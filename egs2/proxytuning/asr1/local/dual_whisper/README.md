# Whisper-small 冻结 + Whisper-tiny 全参数梯度侧微调

本实验仅融合 logits：`alpha * logits_small + (1-alpha) * logits_tiny`，默认 alpha=0.5。交叉熵直接接收 logits，不做预先 softmax，无蒸馏、KL、FFM、Adapter 或新增可训练层。原来的 run.sh 和公共 ESPnet 文件没有修改。

## 已阅读的现有路径与改动原因

- `espnet2/asr/encoder/whisper_encoder.py`：CPU 加载 Whisper 预训练权重并复制 encoder，计算 Whisper log-mel，支持变长位置编码；本实验直接复用。
- `espnet2/asr/decoder/whisper_decoder.py`：加载并复制 decoder；训练返回原始词表 logits，`use_proxy_tuning=True` 时单步评分也返回 logits。现有实现重算完整前缀，没有 KV cache。
- `espnet2/asr/proxy_espnet_model.py`：现有 proxy 训练仅计算 tiny decoder 的 loss，未融合 small decoder logits；eval 的 CER 是 teacher-forcing argmax。
- `espnet2/asr/espnet_model.py`：复用 encode、SOS/EOS、ignore_id、batch 截断、返回 stats/weight 的接口；覆盖 attention loss，支持快速 teacher-forcing 验证和按需自回归验证。
- `espnet2/tasks/asr.py` / `tasks/abs_task.py`：复用模型选择、preprocessor、数据加载、optimizer 和 scheduler 构造；原 optimizer 默认包含所有参数，新增 Task 只暴露 tiny 参数。
- `espnet2/train/trainer.py`：保留 AMP/GradScaler、累积、裁剪、更新、日志、checkpoint、resume 和 n-best averaging；薄子类增加训练 epoch CUDA 峰值统计。
- `espnet2/bin/asr_inference.py` / `espnet/nets/{batch_beam_search,beam_search}.py`：复用完整推理与评分入口，更换模型 Task 和 decoder scorer。
- `text/whisper_token_id_converter.py` / `train/preprocessor.py`：确认预处理器已经在转录前加入 zh/transcribe/notimestamps，训练不重复添加。

公共源码已有用户修改，而且在当前可写工作区之外，所以采用 recipe 内可选入口。单模型通过原 run.sh 使用原来的路径。

## 修改文件

| 文件 | 作用 |
| --- | --- |
| `local/dual_whisper/model.py` | 冻结/eval small，两路独立 encoder/decoder、logits 融合、CE、自回归 CER、轻量 state_dict |
| `local/dual_whisper/task.py` | 可选 dual_whisper 模型注册、词表/语言/特殊 token 检查、只优化 tiny |
| `local/dual_whisper/trainer.py` | 原 Trainer 子类，记录完整训练 epoch 的 CUDA peak allocated/reserved（字节） |
| `local/dual_whisper/train.py`、`inference.py`、`__init__.py` | 可选训练与标准推理入口 |
| `local/dual_python.sh` | 为 asr.sh 的 Python 模块命令选择本地入口；其他模块照常执行 |
| `run_dual.sh` | 复用 asr.sh stages 11–13，独立实验目录 |
| `conf/tuning/train_asr_whisper_dual.yaml` | small/tiny、alpha、冻结开关、CE、AMP、优化器等配置 |
| `conf/whisper_decode_dual.yaml` | 最终自回归解码 beam=3、输出上限=128、CTC/LM 权重=0 |
| `local/dual_whisper/check.py` | 真实预训练模型的梯度、融合公式、loss 下降、shape、评分一致性及缺失权重检查 |
| `local/smoke_dual.sh`、`local/dual_whisper/inspect_smoke.py` | 真实数据少量 batch、累积、恢复、独立推理、checkpoint/optimizer/scheduler 检查 |
| `local/dual_whisper/check_validation.py` | 真实数据梯度、TF 无搜索、原错误计算器一致性、可选 AR 模式回归检查 |
| 本文件 | 实现说明、运行方法及限制 |

## 关键行为

1. small 的 encoder 和 decoder 都设置 requires_grad=False；各自前向包在 no_grad 中。包装器覆盖 train()，无论 model.train() 或 model.eval()，small 始终 eval。
2. tiny encoder/decoder 的全部参数正常训练。optimizer 参数集合与 requires_grad=True 的参数集合精确一致。
3. 两路 encoder 分别产生 768/384 维表示。为兼容 ESPnet 基于单个 tensor 的 beam 接口，沿最后一维拼接作为**运输容器**；decoder 立即 split，各自仅消费自己的表示。没有相加、投影、交叉特征融合。
4. 两路 decoder 使用完全相同的 ys_in，检查 logits shape 相同后融合。训练中不 detach tiny/logits_final/loss；ESPnet 原有日志仍会 detach stats 副本。
5. add_sos_eos、EOS 监督、padding/ignore_id 保留。预处理已有的三个提示 token 作为输入，按当前 proxy 实现从 loss target 屏蔽；真实文本和 EOS 参与 CE。默认 sum CE / batch_size；length_normalized_loss=true 时除以有效监督 token 数。
6. 检查全部 51,865 个 token 的顺序，兼容 ESPnet 导出词表将 Ġ 替换成空格的规则，以及读取时的 rstrip；强制 multilingual small/tiny、zh/transcribe、原生 SOS/EOS。
7. 每步先融合原始 next-token logits，再 log_softmax 给 ESPnet beam scorer；两个模型不会独立生成文本。serial/batch scorer 均支持；两路 state 为 `(None, None)`，与当前底层无缓存实现一致。
8. 默认 `validation_decode_mode: teacher_forcing`：每轮 valid loss/acc 使用融合 logits，argmax 后去掉固定提示位置，再直接使用原 ESPnet ErrorCalculator 的词表转换、padding 截断、space/blank 处理和编辑距离计算。该计算器没有额外的中文规范化器，不添加新的规范化协议。错误率只命名为 `cer_tf/wer_tf`，不输出 `cer/wer`；无 BeamSearch 调用。默认选模和早停均用 `valid.loss/min`。`validation_decode_mode: autoregressive` 则保留完整融合式 beam 验证，输出 `cer/wer`，不输出 `cer_tf/wer_tf`，默认仍按 loss 选模。验证统计沿用原 ESPnet 的 batch size 权重，不声称是精确语料级错误率；最终语料级真实 CER 用 stage 13 的评分结果。
9. 保存的权重只有 tiny 的参数和必要 buffer；small 的参数、buffer 均不保存。恢复先通过构造器加载 small 的原始预训练权重，再恢复 tiny。缺少 tiny 键即报错，即使上游以 strict=False 加载。Trainer 照常保存/恢复 optimizer、scheduler、reporter、AMP scaler。
10. 保留实验目录的 `config.yaml` 与 checkpoint 配套使用，配置含完整词表、模型名称、alpha、语言、任务及训练设置；预训练缓存中的 small.pt/tiny.pt 必须可用。不能仅把 tiny 权重交给原单模型推理入口。

## 运行

在本 recipe 目录，激活项目环境：

```bash
source ./path.sh
# 已有 dump/raw 数据和 asr_stats_raw_zh_whisper_multilingual，可直接从训练开始。
./run_dual.sh
```

默认保存到 `exp/asr_whisper_dual_small_tiny`，避免覆盖现有实验。中断后重复命令会使用 ESPnet 的 resume；选模权重现在为 `valid.loss.ave.pth`。只做解码/最终 CER：

```bash
./run_dual.sh --stage 12 --stop_stage 13
```

单独运行梯度与显存检查（固定合成波形和中文标签，只验证功能）：

```bash
PYTHONPATH=local python3 -m dual_whisper.check --device cuda --amp \
    --output exp/dual_whisper_checks_cuda.json
# CPU 功能检查
OMP_NUM_THREADS=2 PYTHONPATH=local python3 -m dual_whisper.check
```

真实数据少量训练、恢复、独立解码检查：

```bash
# 2 个真实训练 utterance，1 个验证 utterance，accum_grad=2；默认快速验证，最终解码临时上限 8 token。
# 使用独立 exp/dual_whisper_smoke，首次测试应使用新的该目录。
DUAL_SMOKE_DIR=exp/dual_whisper_smoke_tf ./local/smoke_dual.sh --ngpu 0 --use_amp false
# GPU AMP 测试：
DUAL_SMOKE_NGPU=1 ./local/smoke_dual.sh --ngpu 1 --use_amp true
```

可用 `DUAL_PYTHON=/path/to/python` 指定解释器。smoke 的命令行参数用于训练，`DUAL_SMOKE_NGPU` 用于独立推理。小样本测试不是质量评估。完整质量实验应使用正式 run_dual.sh 和 stage 13，不使用 8 token 上限。

配置核心部分（完整配置见 conf）：

```yaml
model: dual_whisper
model_conf:
    dual_model: true
    small_model_name: small
    tiny_model_name: tiny
    fusion_alpha: 0.5
    freeze_small: true
    train_tiny: true
    use_ffm: false
    validation_decode_mode: teacher_forcing
    ctc_weight: 0.0
    lsm_weight: 0.0
    sym_sos: <|startoftranscript|>
    sym_eos: <|endoftext|>
init: null
use_amp: true
```

不要将 init 设置成随机初始化方法。标准单模型请继续使用原 run.sh/原配置；dual_whisper 模式会拒绝 FFM、adapter、CTC、label smoothing 或额外冻结选项。

## 当前实测

- CPU，真实缓存的 small/tiny 预训练权重；small 总参数 240,582,912，可训练 0；tiny 总参数及可训练参数均为 37,184,640。
- forward/backward：small 所有梯度 None；tiny 全部参数的梯度存在且有限，主要卷积、embedding、self/cross-attention query 权重梯度非零；small 输出无计算图，tiny/final 输出有计算图。
- 固定合成 batch 3 次更新，loss 10.8249 → 4.2249 → 1.0630，更新后 0.1542。仅证明能优化此 batch。
- 训练 logits 为 `[1, 5, 51865]`，单步评分为 `[1, 51865]`；确认 0.5:0.5 公式和 serial/batch next-token log probabilities 一致。
- 真实数据完成第一 epoch（2 batch），恢复后完成第二 epoch（2 batch），optimizer step=2、scheduler last_epoch=2；tiny 的两路模块均继续更新。
- 单模型权重文件 151,121,721 字节（约 144 MiB）；checkpoint 448,744,855 字节（约 428 MiB，含 AdamW 状态）；167 个模型键，无 small 权重。
- 标准独立推理入口运行成功，使用 BatchBeamSearch。测试结果和细节见 `exp/dual_whisper_checks.json`、`exp/dual_whisper_smoke/checkpoint_checks.json` 及同目录日志。
- 当前环境 CUDA 不可用，NVIDIA 驱动无法连接。FP16/GradScaler 路径已接入但未在 GPU 实测；peak allocated/reserved 在 CPU 报告为 null。尚无训练显存下降比例或完整验证/测试集 CER 结论。GPU 测量时应保持 batch、长度、AMP、累积和 optimizer 相同，并与原双模型反传方案比较；训练日志每个 epoch 记录 CUDA 峰值。

第一版刻意保留无 KV cache 的现有解码实现；长输出解码可能较慢。若正式数据需要超过 128 新 token，应同时修改 model_conf.validation_max_tokens 和推理配置 maxlenratio 的绝对值，且不超过 Whisper decoder 上下文容量扣除提示 token 后的长度。

## 快速验证回归检查

```bash
OMP_NUM_THREADS=2 PYTHONPATH=local python3 -m dual_whisper.check_validation
```

该检查使用现有 smoke 子集中的两条真实语音（含不同标签长度和 padding），执行一次训练 forward/backward，检查 small 无梯度、tiny 全部参数梯度有限且主要权重非零；TF 验证期间将 BeamSearch 替换为抛错函数，确认完全不触发搜索，且 cer_tf/wer_tf 与原 ErrorCalculator 一致；再验证按需 autoregressive 模式及其指标名称。完整 Trainer/恢复/独立推理测试使用 smoke 脚本，报告位于所选 DUAL_SMOKE_DIR，默认配置的训练和验证日志不应含 beam_search 输出，最终 inference.log 则正常包含。

按需全量自回归验证（仍按 loss 选模）：

```bash
./run_dual.sh --asr_exp exp/asr_whisper_dual_ar \
  --asr_args "--model_conf validation_decode_mode=autoregressive"
```

如果自行改为按自回归 CER 选模，须同时修改 best_model_criterion、early_stopping_criterion 和 inference_asr_model；TF 模式不得使用 `valid.cer` 选模。更新源码不会改变已经运行的 Python 进程，需正常停止后重新启动训练才能采用新验证方式；不修改历史实验目录中的配置和日志。
