"""Trainer module."""
import json
from collections import Counter
import os
#
import argparse
import dataclasses
import logging
import time
from contextlib import contextmanager
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import humanfriendly
import numpy as np
import torch
import torch.nn
import torch.optim
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsEpochStepScheduler,
    AbsScheduler,
    AbsValEpochStepScheduler,
)
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.kwargs2args import kwargs2args

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

autocast_args = dict()
if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import GradScaler, autocast

    if (
        V(torch.__version__) >= V("1.10.0")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        autocast_args = dict(dtype=torch.bfloat16)
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

    GradScaler = None

try:
    import fairscale
except ImportError:
    fairscale = None

try:
    import loralib as lora
except Exception:
    lora = None
from loralib.layers import MoELoRALinear  
try:
    import s3prl # type: ignore
except Exception:
    s3prl = None


@dataclasses.dataclass
class TrainerOptions:
    ngpu: int
    resume: bool
    use_amp: bool
    train_dtype: str
    grad_noise: bool
    accum_grad: int
    grad_clip: float
    grad_clip_type: float
    log_interval: Optional[int]
    no_forward_run: bool
    use_matplotlib: bool
    use_tensorboard: bool
    use_wandb: bool
    adapter: str
    use_adapter: bool
    save_strategy: str
    output_dir: Union[Path, str]
    max_epoch: int
    seed: int
    sharded_ddp: bool
    patience: Optional[int]
    keep_nbest_models: Union[int, List[int]]
    nbest_averaging_interval: int
    early_stopping_criterion: Sequence[str]
    best_model_criterion: Sequence[Sequence[str]]
    val_scheduler_criterion: Sequence[str]
    unused_parameters: bool
    wandb_model_log_interval: int
    create_graph_in_tensorboard: bool
    gradient_as_bucket_view: bool
    ddp_comm_hook: Optional[str]


class Trainer:
    """Trainer having a optimizer.

    If you'd like to use multiple optimizers, then inherit this class
    and override the methods if necessary - at least "train_one_epoch()"

    >>> class TwoOptimizerTrainer(Trainer):
    ...     @classmethod
    ...     def add_arguments(cls, parser):
    ...         ...
    ...
    ...     @classmethod
    ...     def train_one_epoch(cls, model, optimizers, ...):
    ...         loss1 = model.model1(...)
    ...         loss1.backward()
    ...         optimizers[0].step()
    ...
    ...         loss2 = model.model2(...)
    ...         loss2.backward()
    ...         optimizers[1].step()

    """

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @typechecked
    def build_options(cls, args: argparse.Namespace) -> TrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()"""
        return build_dataclass(TrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Reserved for future development of another Trainer"""
        pass

    @staticmethod
    def resume(
        checkpoint: Union[str, Path],
        model: torch.nn.Module,
        reporter: Reporter,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler], # type: ignore
        ngpu: int = 0,
        strict: bool = True,
    ):
        states = torch.load(
            checkpoint,
            map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
            weights_only=False,
        )
        model.load_state_dict(states["model"], strict=strict)
        reporter.load_state_dict(states["reporter"])
        for optimizer, state in zip(optimizers, states["optimizers"]):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, states["schedulers"]):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is None:
                logging.warning("scaler state is not found")
            else:
                scaler.load_state_dict(states["scaler"])

        logging.info(f"The training was resumed using {checkpoint}")
    
    @classmethod
    @typechecked
    def run(
        cls,
        model: AbsESPnetModel,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        train_iter_factory: AbsIterFactory,
        valid_iter_factory: AbsIterFactory,
        plot_attention_iter_factory: Optional[AbsIterFactory],
        trainer_options,
        distributed_option: DistributedOption,
    ) -> None:
        """Perform training. This method performs the main process of training."""
        # NOTE(kamo): Don't check the type more strictly as far trainer_options
        assert is_dataclass(trainer_options), type(trainer_options)
        assert len(optimizers) == len(schedulers), (len(optimizers), len(schedulers))

        if isinstance(trainer_options.keep_nbest_models, int):
            keep_nbest_models = [trainer_options.keep_nbest_models]
        else:
            if len(trainer_options.keep_nbest_models) == 0:
                logging.warning("No keep_nbest_models is given. Change to [1]")
                trainer_options.keep_nbest_models = [1]
            keep_nbest_models = trainer_options.keep_nbest_models

        output_dir = Path(trainer_options.output_dir)
        reporter = Reporter()
        if trainer_options.use_amp:
            if V(torch.__version__) < V("1.6.0"):
                raise RuntimeError(
                    "Require torch>=1.6.0 for  Automatic Mixed Precision"
                )
            if trainer_options.sharded_ddp:
                if fairscale is None:
                    raise RuntimeError(
                        "Requiring fairscale. Do 'pip install fairscale'"
                    )
                scaler = fairscale.optim.grad_scaler.ShardedGradScaler()
            else:
                scaler = GradScaler()
        else:
            scaler = None

        adapter = getattr(trainer_options, "adapter", None)
        use_adapter = getattr(trainer_options, "use_adapter", False)
        save_strategy = getattr(trainer_options, "save_strategy", "all")
        if use_adapter:
            if adapter == "lora" and lora is None:
                raise RuntimeError("Requiring loralib. Do 'pip install loralib'")
            elif adapter == "houlsby" and s3prl is None:
                print("Error: S3PRL is not properly installed.")
                print("Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done")
                raise RuntimeError("Requiring S3PRL. ")

        if trainer_options.resume and (output_dir / "checkpoint.pth").exists():
            cls.resume(
                checkpoint=output_dir / "checkpoint.pth",
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                reporter=reporter,
                scaler=scaler,
                ngpu=trainer_options.ngpu,
                strict=not use_adapter,
            )

        start_epoch = reporter.get_epoch() + 1
        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )
        # # --- GoRA 新增开始：仅在训练开始前（首次epoch）执行一次初始化 ---=========================================
        # # 仅在 rank 0 执行（分布式训练）+ 非恢复训练（resume）+ 使用lora适配器时执行
        # if (start_epoch == 1 and adapter == "gora" and use_adapter and (not distributed_option.distributed or distributed_option.dist_rank == 0)):
        #     try:
        #         from loralib.layers import GoRALinear, allocate_ranks_by_importance 
        #         from loralib.utils import mark_only_gora_as_trainable 
        #         gora_cfg = {
        #             "total_param_budget": getattr(trainer_options, "gora_total_param_budget", 19464192),#6488064
        #             "grad_steps": getattr(trainer_options, "gora_grad_steps", 4),
        #             "min_rank": getattr(trainer_options, "gora_min_rank", 36),
        #             "max_rank": getattr(trainer_options, "gora_max_rank", 64),
        #             "importance_type": getattr(trainer_options, "gora_importance_type", "union_mean"),
        #             "stable_gamma": getattr(trainer_options, "gora_stable_gamma", 16.),
        #         }
        #         logging.info(f"Start GoRA initialization with config: {gora_cfg}")
        #         if trainer_options.use_amp:
        #             if trainer_options.sharded_ddp and fairscale is not None:
        #                 local_scaler = fairscale.optim.grad_scaler.ShardedGradScaler()
        #             else:
        #                 local_scaler = torch.cuda.amp.GradScaler()
        #         else:
        #             local_scaler = None
        #         model.train()
        #         for name, module in model.named_modules():
        #             if isinstance(module, GoRALinear) and module.r > 0:
        #                 module.grad_stored = None  # 清空累积梯度
        #                 module.grad_steps = 0      # 清空步数
        #                 module.name = name         # 记录层名
        #                 if module.r > 0:
        #                     module.weight.requires_grad = True
        #         train_iter = train_iter_factory.build_iter(start_epoch)  
        #         grad_steps_completed = 0
        #         grad_accum_dict = {}
        #         for name, module in model.named_modules():
        #             if isinstance(module, GoRALinear) and module.r > 0:
        #                 grad_accum_dict[name] = None  # 初始化为None
        #         for iiter, (utt_id, batch) in enumerate(train_iter):
        #             if grad_steps_completed >= gora_cfg["grad_steps"]:
        #                 break  # 累积4个batch后停止
        #             batch["utt_id"] = utt_id
        #             batch = to_device(batch, "cuda" if trainer_options.ngpu > 0 else "cpu")
                    
        #             with torch.cuda.amp.autocast(enabled=trainer_options.use_amp):
        #                 retval = model(** batch)
        #                 loss = retval["loss"] if isinstance(retval, dict) else retval[0]
        #                 loss = loss / gora_cfg["grad_steps"]  # 均分loss，避免梯度爆炸
        #             logging.info(f"=== GoRA Grad Accum Batch {grad_steps_completed+1}/{gora_cfg['grad_steps']} ===")
        #             logging.info(f"  Loss: {loss.item():.6f}, NaN: {torch.isnan(loss).item()}, Inf: {torch.isinf(loss).item()}")
        #             if trainer_options.use_amp and local_scaler is not None:
        #                 local_scaler.scale(loss).backward()
        #             else:
        #                 loss.backward()
        #             for name, module in model.named_modules():
        #                 if isinstance(module, GoRALinear) and module.r > 0:
        #                     if module.weight.grad is None:
        #                         logging.warning(f"⚠️  Layer {name} has no weight gradient in batch {grad_steps_completed+1}")
        #                         continue
        #                     current_grad = module.weight.grad.detach().clone()
        #                     current_grad = torch.nan_to_num(current_grad, nan=1e-6, posinf=1e6, neginf=-1e6)
        #                     if grad_accum_dict[name] is None:
        #                         grad_accum_dict[name] = current_grad
        #                     else:
        #                         grad_accum_dict[name] += current_grad
        #                     module.grad_steps += 1
            
        #             grad_steps_completed += 1
        #             logging.info(f"✅ Batch {grad_steps_completed} gradient accumulated")
        #         if local_scaler is not None:
        #             del local_scaler
        #         for name, module in model.named_modules():
        #             if isinstance(module, GoRALinear) and module.r > 0:
        #                 if grad_accum_dict[name] is None:
        #                     logging.warning(f"⚠️  Layer {name} has no accumulated gradient, use default init")
        #                     module.grad_stored = None
        #                     continue

        #                 grad_mean = grad_accum_dict[name] / gora_cfg["grad_steps"]
        #                 # grad_mean *= gora_cfg["grad_amp_factor"]
        #                 grad_mean *= 1   #e4
        #                 grad_mean = torch.where(torch.norm(grad_mean) < 1e-8, 
        #                                        torch.ones_like(grad_mean) * 1e-6, 
        #                                        grad_mean)
                
        #                 module.grad_stored = grad_mean
        #                 grad_norm = torch.norm(module.grad_stored).item()
        #                 grad_abs_mean = torch.mean(torch.abs(module.grad_stored)).item()
        #                 logging.info(f"📊 Layer {name} - Grad Norm: {grad_norm:.6f}, Abs Mean: {grad_abs_mean:.6f}")
        #         for name, module in model.named_modules():
        #             if isinstance(module, GoRALinear) and module.r > 0:
        #                 if module.grad_steps != gora_cfg["grad_steps"]:
        #                     logging.warning(f"⚠️  Layer {name} grad steps mismatch: actual={module.grad_steps}, expected={gora_cfg['grad_steps']}")

        #         named_ranks = allocate_ranks_by_importance(
        #             model=model,
        #             total_param_budget=gora_cfg["total_param_budget"],  # 替换为参数量预算
        #             importance_type=gora_cfg["importance_type"],
        #             min_rank=gora_cfg["min_rank"],
        #             max_rank=gora_cfg["max_rank"],
        #             param_tolerance=0.05,
        #         )
        #         logging.info(f"GoRA rank allocation result: {named_ranks}")
        #         # for name, module in model.named_modules():
        #         #     if isinstance(module, GoRALinear) and name in named_ranks:
        #         #         module.dynamic_init(
        #         #             target_rank=named_ranks[name],
        #         #             stable_gamma=gora_cfg["stable_gamma"]
        #         #         )
        #         scaler = None
        #         optimizers[0].zero_grad(set_to_none=True)

        #         mark_only_gora_as_trainable(model, bias="none") 
        #         for optimizer in optimizers:
        #             optimizer.zero_grad(set_to_none=True)
        #         torch.cuda.empty_cache()
        #         # # ====================== 新增：保存grad_G（极简版） ======================
        #         # gradG_all = {}
        #         # for name, module in model.named_modules():
        #         #     if isinstance(module, GoRALinear) and module.grad_stored is not None:
        #         #         gradG_all[name] = module.grad_stored.half()  # float16压缩
        #         # torch.save(gradG_all, output_dir / "all_gradG.pt")
        #         # logging.info(f"grad_G合并保存到 {output_dir / 'all_gradG.pt'}，共{len(gradG_all)}层")
        #         # # ====================== 新增结束 ======================
        #         with open(output_dir / "gora_rank_allocation.json", "w") as f:
        #             json.dump(named_ranks, f, indent=2)
        #         logging.info(f"GoRA initialization finished, rank saved to {output_dir / 'gora_rank_allocation.json'}")
        #     except Exception as e:
        #         logging.error(f"GoRA initialization failed: {e}", exc_info=True)
        #         raise RuntimeError("GoRA init error, stop training")
        # # --- GoRA 新增结束 ---================================================================
        if distributed_option.distributed:
            if trainer_options.sharded_ddp:
                dp_model = fairscale.nn.data_parallel.ShardedDataParallel(
                    module=model,
                    sharded_optimizer=optimizers,
                )
            else:
                dp_model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=(
                        # Perform multi-Process with multi-GPUs
                        [torch.cuda.current_device()]
                        if distributed_option.ngpu == 1
                        # Perform single-Process with multi-GPUs
                        else None
                    ),
                    output_device=(
                        torch.cuda.current_device()
                        if distributed_option.ngpu == 1
                        else None
                    ),
                    find_unused_parameters=trainer_options.unused_parameters,
                    gradient_as_bucket_view=trainer_options.gradient_as_bucket_view,
                )

                # Register DDP communication hook
                if trainer_options.ddp_comm_hook is not None:
                    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (  # noqa: E501
                        bf16_compress_hook,
                        fp16_compress_hook,
                    )

                    _hooks = {
                        "fp16_compress_hook": fp16_compress_hook,
                        "bf16_compress_hook": bf16_compress_hook,
                    }
                    dp_model.register_comm_hook(
                        None,
                        _hooks[trainer_options.ddp_comm_hook],
                    )
                    logging.info(
                        f"Registered DDP communication hook: "
                        f"{trainer_options.ddp_comm_hook}"
                    )

        elif distributed_option.ngpu > 1:
            dp_model = torch.nn.parallel.DataParallel(
                model,
                device_ids=list(range(distributed_option.ngpu)),
            )
        else:
            # NOTE(kamo): DataParallel also should work with ngpu=1,
            # but for debuggability it's better to keep this block.
            dp_model = model

        if trainer_options.use_tensorboard and (
            not distributed_option.distributed or distributed_option.dist_rank == 0
        ):
            from torch.utils.tensorboard import SummaryWriter

            train_summary_writer = SummaryWriter(
                str(output_dir / "tensorboard" / "train")
            )
            valid_summary_writer = SummaryWriter(
                str(output_dir / "tensorboard" / "valid")
            )
        else:
            train_summary_writer = None
        
        start_time = time.perf_counter()
        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            if iepoch != start_epoch:
                logging.info(
                    "{}/{}epoch started. Estimated time to finish: {}".format(
                        iepoch,
                        trainer_options.max_epoch,
                        humanfriendly.format_timespan(
                            (time.perf_counter() - start_time)
                            / (iepoch - start_epoch)
                            * (trainer_options.max_epoch - iepoch + 1)
                        ),
                    )
                )
            else:
                logging.info(f"{iepoch}/{trainer_options.max_epoch}epoch started")
            set_all_random_seed(trainer_options.seed + iepoch)

            reporter.set_epoch(iepoch)
            # 1. Train and validation for one-epoch
            torch.cuda.empty_cache()
            with reporter.observe("train") as sub_reporter:
                all_steps_are_invalid = cls.train_one_epoch(
                    model=dp_model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    scaler=scaler,
                    summary_writer=train_summary_writer,
                    options=trainer_options,
                    distributed_option=distributed_option,
                )

            torch.cuda.empty_cache()
            with reporter.observe("valid") as sub_reporter:
                cls.validate_one_epoch(
                    model=dp_model,
                    iterator=valid_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                    distributed_option=distributed_option,
                )

            torch.cuda.empty_cache()
            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # att_plot doesn't support distributed
                if plot_attention_iter_factory is not None:
                    with reporter.observe("att_plot") as sub_reporter:
                        cls.plot_attention(
                            model=model,
                            output_dir=output_dir / "att_ws",
                            summary_writer=train_summary_writer,
                            iterator=plot_attention_iter_factory.build_iter(iepoch),
                            reporter=sub_reporter,
                            options=trainer_options,
                        )

            # 2. LR Scheduler step
            for scheduler in schedulers:
                if isinstance(scheduler, AbsValEpochStepScheduler):
                    scheduler.step(
                        reporter.get_value(*trainer_options.val_scheduler_criterion)
                    )
                elif isinstance(scheduler, AbsEpochStepScheduler):
                    scheduler.step()
            if trainer_options.sharded_ddp:
                for optimizer in optimizers:
                    if isinstance(optimizer, fairscale.optim.oss.OSS):
                        optimizer.consolidate_state_dict()

            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # 3. Report the results
                logging.info(reporter.log_message())
                if trainer_options.use_matplotlib:
                    reporter.matplotlib_plot(output_dir / "images")
                if train_summary_writer is not None:
                    reporter.tensorboard_add_scalar(train_summary_writer, key1="train")
                    reporter.tensorboard_add_scalar(valid_summary_writer, key1="valid")
                if trainer_options.use_wandb:
                    reporter.wandb_log()

                # 4. Save/Update the checkpoint
                model_state_dict = model.state_dict()
                if use_adapter:
                    if save_strategy == "all":
                        model_state_dict = model_state_dict
                    elif save_strategy == "adapter_only":
                        if adapter == "lora":
                            model_state_dict = lora.lora_state_dict(model)
                        elif adapter == "houlsby":
                            model_state_dict = {
                                k: v
                                for k, v in model_state_dict.items()
                                if "adapter" in k
                            }
                        else:
                            raise ValueError(f"Adapter type {adapter} not supported")
                    else:  # save_strategy == "required_grad_only"
                        for n, p in model.named_parameters():
                            if not p.requires_grad:
                                model_state_dict.pop(n)

                torch.save(
                    {
                        "model": model_state_dict,
                        "reporter": reporter.state_dict(),
                        "optimizers": [o.state_dict() for o in optimizers],
                        "schedulers": [
                            s.state_dict() if s is not None else None
                            for s in schedulers
                        ],
                        "scaler": scaler.state_dict() if scaler is not None else None,
                    },
                    output_dir / "checkpoint.pth",
                )

                # 5. Save and log the model and update the link to the best model
                torch.save(model_state_dict, output_dir / f"{iepoch}epoch.pth")

                # Creates a sym link latest.pth -> {iepoch}epoch.pth
                p = output_dir / "latest.pth"
                if p.is_symlink() or p.exists():
                    p.unlink()
                p.symlink_to(f"{iepoch}epoch.pth")

                _improved = []
                for _phase, k, _mode in trainer_options.best_model_criterion:
                    # e.g. _phase, k, _mode = "train", "loss", "min"
                    if reporter.has(_phase, k):
                        best_epoch = reporter.get_best_epoch(_phase, k, _mode)
                        # Creates sym links if it's the best result
                        if best_epoch == iepoch:
                            p = output_dir / f"{_phase}.{k}.best.pth"
                            if p.is_symlink() or p.exists():
                                p.unlink()
                            p.symlink_to(f"{iepoch}epoch.pth")
                            _improved.append(f"{_phase}.{k}")
                if len(_improved) == 0:
                    logging.info("There are no improvements in this epoch")
                else:
                    logging.info(
                        "The best model has been updated: " + ", ".join(_improved)
                    )

                log_model = (
                    trainer_options.wandb_model_log_interval > 0
                    and iepoch % trainer_options.wandb_model_log_interval == 0
                )
                if log_model and trainer_options.use_wandb:
                    import wandb

                    logging.info("Logging Model on this epoch :::::")
                    artifact = wandb.Artifact(
                        name=f"model_{wandb.run.id}",
                        type="model",
                        metadata={"improved": _improved},
                    )
                    artifact.add_file(str(output_dir / f"{iepoch}epoch.pth"))
                    aliases = [
                        f"epoch-{iepoch}",
                        "best" if best_epoch == iepoch else "",
                    ]
                    wandb.log_artifact(artifact, aliases=aliases)

                # 6. Remove the model files excluding n-best epoch and latest epoch
                _removed = []
                # Get the union set of the n-best among multiple criterion
                nbests = set().union(
                    *[
                        set(reporter.sort_epochs(ph, k, m)[: max(keep_nbest_models)])
                        for ph, k, m in trainer_options.best_model_criterion
                        if reporter.has(ph, k)
                    ]
                )

                # Generated n-best averaged model
                if (
                    trainer_options.nbest_averaging_interval > 0
                    and iepoch % trainer_options.nbest_averaging_interval == 0
                ):
                    average_nbest_models(
                        reporter=reporter,
                        output_dir=output_dir,
                        best_model_criterion=trainer_options.best_model_criterion,
                        nbest=keep_nbest_models,
                        suffix=f"till{iepoch}epoch",
                    )

                for e in range(1, iepoch):
                    p = output_dir / f"{e}epoch.pth"
                    if p.exists() and e not in nbests:
                        p.unlink()
                        _removed.append(str(p))
                if len(_removed) != 0:
                    logging.info("The model files were removed: " + ", ".join(_removed))

            # 7. If any updating haven't happened, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    "The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if trainer_options.patience is not None:
                if reporter.check_early_stopping(
                    trainer_options.patience, *trainer_options.early_stopping_criterion
                ):
                    break

        else:
            logging.info(
                f"The training was finished at {trainer_options.max_epoch} epochs "
            )

        # Generated n-best averaged model
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=trainer_options.best_model_criterion,
                nbest=keep_nbest_models,
            )
      

    @classmethod
    @typechecked
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler], # type: ignore
        reporter: SubReporter,
        summary_writer,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> bool:
        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        create_graph_in_tensorboard = options.create_graph_in_tensorboard
        distributed = distributed_option.distributed

        # # -------------------------- 新增：裁剪相关计数器（硬编码） --------------------------
        # step_counter = 0  # 训练步数计数器
        # # -------------------------- 新增：裁剪相关计数器（硬编码） ------
        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        for iiter, (utt_id, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch["utt_id"] = utt_id

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

            if no_forward_run:
                all_steps_are_invalid = False
                continue

            if (
                create_graph_in_tensorboard
                and iiter == 1
                and summary_writer is not None
            ):
                if distributed:
                    _model = getattr(model, "module")
                else:
                    _model = model
                    if _model is not None:
                        try:
                            _args = kwargs2args(_model.forward, batch)
                        except (ValueError, TypeError):
                            logging.warning(
                                "inpect.signature() is failed for the model. "
                                "The graph can't be added for tensorboard."
                            )
                        else:
                            try:
                                summary_writer.add_graph(
                                    _model, _args, use_strict_trace=False
                                )
                            except Exception:
                                logging.warning(
                                    "summary_writer.add_graph() "
                                    "is failed for the model. "
                                    "The graph can't be added for tensorboard."
                                )
                            del _args
                    else:
                        logging.warning(
                            "model.module is not found (This should be a bug.)"
                        )
                del _model

            with autocast(
                scaler is not None,
                **autocast_args,
            ):
                with reporter.measure_time("forward_time"):
                    retval = model(**batch)
                   

                    # Note(kamo):
                    # Supporting two patterns for the returned value from the model
                    #   a. dict type
                    if isinstance(retval, dict):
                        loss = retval["loss"]
                        stats = retval["stats"]
                        weight = retval["weight"]
                        optim_idx = retval.get("optim_idx")
                        if optim_idx is not None and not isinstance(optim_idx, int):
                            if not isinstance(optim_idx, torch.Tensor):
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {type(optim_idx)}"
                                )
                            if optim_idx.dim() >= 2:
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {optim_idx.dim()}dim tensor"
                                )
                            if optim_idx.dim() == 1:
                                for v in optim_idx:
                                    if v != optim_idx[0]:
                                        raise RuntimeError(
                                            "optim_idx must be 1dim tensor "
                                            "having same values for all entries"
                                        )
                                optim_idx = optim_idx[0].item()
                            else:
                                optim_idx = optim_idx.item()

                    #   b. tuple or list type
                    else:
                        loss, stats, weight = retval
                        optim_idx = None

                    retval = None

                stats = {k: v for k, v in stats.items() if v is not None}
                if ngpu > 1 or distributed:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()

                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed)

                    # Now weight is summation over all workers
                    loss /= weight
                if distributed:
                    # NOTE(kamo): Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= torch.distributed.get_world_size()

                loss /= accum_grad
                # # -------------------------- 新增：添加AdaLoRA正交正则化损失 --------------------------
                # # 遍历模型所有层，累加正交正则化损失
                # ada_reg_loss = 0.0
                # # 兼容DDP：取model.module（分布式下model是DDP包装的）
                # _model = model.module if hasattr(model, 'module') else model
                # for module in _model.modules():
                #     if hasattr(module, 'ortho_reg'):
                #         ada_reg_loss += module.ortho_reg()
                # # 加到总损失
                # loss += 0.005 * ada_reg_loss#loss += gamma * ada_reg_loss
                # # -------------------------- 新增：添加AdaLoRA正交正则化损失 --------------------------

            reporter.register(stats, weight)

            with reporter.measure_time("backward_time"):
                
                # load_balance_loss = torch.tensor(0.0, device=loss.device)
                # for module in model.modules(): 
                #     if isinstance(module, MoELoRALinear):
                #         load_balance_loss += module.get_load_balance_loss()
                # loss = loss + load_balance_loss  # 覆盖原loss变量
                if scaler is not None:
                    # Scales loss.  Calls backward() on scaled loss
                    # to create scaled gradients.
                    # Backward passes under autocast are not recommended.
                    # Backward ops run in the same dtype autocast chose
                    # for corresponding forward ops.
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            del loss

            if iiter % accum_grad == 0:
                if scaler is not None:
                    # Unscales the gradients of optimizer's assigned params in-place
                    for iopt, optimizer in enumerate(optimizers):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        scaler.unscale_(optimizer)

                # gradient noise injection
                if grad_noise:
                    add_gradient_noise(
                        model,
                        reporter.get_total_count(),
                        duration=100,
                        eta=1.0,
                        scale_factor=0.55,
                    )

                # compute the gradient norm to check if it is normal or not
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                    norm_type=grad_clip_type,
                )
                # PyTorch<=1.4, clip_grad_norm_ returns float value
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(grad_norm)

                if not torch.isfinite(grad_norm):
                    logging.warning(
                        f"The grad norm is {grad_norm}. Skipping updating the model."
                    )

                    # Must invoke scaler.update() if unscale_() is used in the iteration
                    # to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # Note that if the gradient has inf/nan values,
                    # scaler.step skips optimizer.step().
                    if scaler is not None:
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            scaler.step(optimizer)
                            scaler.update()

                else:
                    reporter.register(
                        {
                            "grad_norm": grad_norm,
                            "clip": torch.where(
                                grad_norm > grad_clip,
                                grad_norm.new_tensor(100),
                                grad_norm.new_tensor(0),
                            ),
                            "loss_scale": scaler.get_scale() if scaler else 1.0,
                        }
                    )
                    all_steps_are_invalid = False
                    with reporter.measure_time("optim_step_time"):
                        for iopt, (optimizer, scheduler) in enumerate(
                            zip(optimizers, schedulers)
                        ):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            if scaler is not None:
                                # scaler.step() first unscales the gradients of
                                # the optimizer's assigned params.
                                scaler.step(optimizer)
                                # Updates the scale for next iteration.
                                scaler.update()
                            else:
                                optimizer.step()
                            if isinstance(scheduler, AbsBatchStepScheduler):
                                scheduler.step()
                # # -------------------------- 新增代码：和上面的for循环同级缩进（关键） --------------------------
                # step_counter += 1
                # if step_counter % 2000 == 0 and step_counter > 0:
                #     _model = model.module if hasattr(model, 'module') else model
                #     for module in _model.modules():
                #         if hasattr(module, 'prune_lambda'):
                #             module.prune_lambda(k=24)#module.prune_lambda(k=prune_k)
                #     # logging.info(f"AdaLoRA prune at step {step_counter}, keep top-{prune_k} lambda")
                # # -------------------------- 新增代码：和上面的for循环同级缩进（关键） --------------------------
                for iopt, optimizer in enumerate(optimizers):
                    if optim_idx is not None and iopt != optim_idx:
                        continue
                    optimizer.zero_grad()

                # Register lr and train/load time[sec/step],
                # where step refers to accum_grad * mini-batch
                reporter.register(
                    dict(
                        {
                            f"optim{i}_lr{j}": pg["lr"]
                            for i, optimizer in enumerate(optimizers)
                            for j, pg in enumerate(optimizer.param_groups)
                            if "lr" in pg
                        },
                        train_time=time.perf_counter() - start_time,
                    ),
                )
                start_time = time.perf_counter()
            # NOTE(kamo): Call log_message() after next()
            reporter.next()
            if iiter % log_interval == 0:
                logging.info(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()
        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        return all_steps_are_invalid

    @classmethod
    @torch.no_grad()
    @typechecked
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed

        model.eval()
        

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for utt_id, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch["utt_id"] = utt_id

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")


            if no_forward_run:
                continue

            with autocast(
                options.use_amp,
                **autocast_args,
            ):
                retval = model(**batch)

            if isinstance(retval, dict):
                stats = retval["stats"]
                weight = retval["weight"]
            else:
                _, stats, weight = retval
            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)

            reporter.register(stats, weight)
            reporter.next()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

    @classmethod
    @torch.no_grad()
    @typechecked
    def plot_attention(
        cls,
        model: torch.nn.Module,
        output_dir: Optional[Path],
        summary_writer,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        reporter: SubReporter,
        options: TrainerOptions,
    ) -> None:
        import matplotlib

        ngpu = options.ngpu
        no_forward_run = options.no_forward_run

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        model.eval()
        for ids, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            assert len(next(iter(batch.values()))) == len(ids), (
                len(next(iter(batch.values()))),
                len(ids),
            )

            batch["utt_id"] = ids

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            # 1. Forwarding model and gathering all attentions
            #    calculate_all_attentions() uses single gpu only.
            att_dict = calculate_all_attentions(model, batch)

            # 2. Plot attentions: This part is slow due to matplotlib
            for k, att_list in att_dict.items():
                assert len(att_list) == len(ids), (len(att_list), len(ids))
                for id_, att_w in zip(ids, att_list):
                    if isinstance(att_w, torch.Tensor):
                        att_w = att_w.detach().cpu().numpy()

                    if att_w.ndim == 2:
                        att_w = att_w[None]
                    elif att_w.ndim == 4:
                        # In multispkr_asr model case, the dimension could be 4.
                        att_w = np.concatenate(
                            [att_w[i] for i in range(att_w.shape[0])], axis=0
                        )
                    elif att_w.ndim > 4 or att_w.ndim == 1:
                        raise RuntimeError(f"Must be 2, 3 or 4 dimension: {att_w.ndim}")

                    w, h = plt.figaspect(1.0 / len(att_w))
                    fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
                    axes = fig.subplots(1, len(att_w))
                    if len(att_w) == 1:
                        axes = [axes]

                    for ax, aw in zip(axes, att_w):
                        ax.imshow(aw.astype(np.float32), aspect="auto")
                        ax.set_title(f"{k}_{id_}")
                        ax.set_xlabel("Input")
                        ax.set_ylabel("Output")
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                    if output_dir is not None:
                        p = output_dir / id_ / f"{k}.{reporter.get_epoch()}ep.png"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(p)

                    if summary_writer is not None:
                        summary_writer.add_figure(
                            f"{k}_{id_}", fig, reporter.get_epoch()
                        )

                    if options.use_wandb:
                        import wandb

                        wandb.log({f"attention plot/{k}_{id_}": wandb.Image(fig)})
            reporter.next()
