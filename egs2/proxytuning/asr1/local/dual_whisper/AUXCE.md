# 最小损失消融：tiny CE + 0.5 × 融合 CE

本轮按最新要求，只修改训练目标，不对齐独立 FFT 设置。原双模型损失默认 `loss_mode: fusion_ce_only`，旧配置和旧 checkpoint 无需增加字段。

新配置 `conf/tuning/train_asr_whisper_dual_auxce.yaml` 逐项复制原双模型配置，仅新增：

```yaml
model_conf:
    loss_mode: tiny_ce_plus_fusion_ce
    lambda_fusion: 0.5
```

两个模型仍各做一次完整 encoder/decoder forward，小模型 logits 直接复用于两个损失，不重复前向。small 前向仍在 no_grad 内，输出不含梯度；tiny logits 不 detach。两项损失使用同一个 criterion_att 和同一个 ys_out、提示屏蔽、padding/ignore_id 及归一化分母。当前原双模型 criterion_att 是 CrossEntropyLoss(ignore_index=-1, reduction=sum)，lsm_weight=0，length_normalized_loss=false，故两项均除以 batch size。没有替换成新的 F.cross_entropy，也没有新增 label smoothing 或 KL。

训练中 loss/loss_att 为总损失，并额外记录 loss_tiny、loss_fusion、loss_total、acc_tiny、acc_fusion。额外统计使用 ESPnet force_gatherable，日志副本 detach，真实训练梯度保持连接。

**仅训练增加辅助 CE。** eval/验证仍计算原融合 CE 和原 cer_tf/wer_tf，不增加 tiny 验证损失或改变选模。best_model_criterion、早停继续按 valid.loss/min，checkpoint averaging 仍保留原来的设置。训练总 loss 的尺度已经改变，不能把它的绝对大小与旧训练 loss 当作同一目标比较；valid.loss 仍是同一个融合目标。

训练命令（只执行 Stage 11，复用原数据和 stats）：

```bash
source ./path.sh
./run_dual.sh
```

从原始预训练 small/tiny 初始化，不加载已有双模型或独立 FFT checkpoint。独立目录 `exp/asr_whisper_dual_auxce_lambda05`。统一由 run_dual.sh 启动，默认 stage=stop_stage=11，不自动解码 Valid/Test。新目录首次训练从预训练模型初始化；后续重复 Stage 11 遵循 ESPnet 原有 resume 行为。旧 run.sh、旧 YAML 和旧 checkpoint 未修改；当前没有启动完整训练。

单 batch 回归检查（两条真实训练语音，有不同长度和 padding；只在内存做一步 optimizer 更新）：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=local python3 -m dual_whisper.check_auxce
# 有可用 CUDA 时可另行检查 AMP 和此测试 batch 的显存：
OMP_NUM_THREADS=2 PYTHONPATH=local python3 -m dual_whisper.check_auxce --device cuda --amp \
  --output exp/dual_whisper_auxce_checks/results_cuda.json
```

结果在 exp/dual_whisper_auxce_checks/results.json。测试包括损失恒等式、复用原 criterion 和 masking、small 无梯度、tiny 全参数有限梯度及实际参数更新、旧损失一致性、验证损失不变、配置仅有两个损失字段差异。CPU 测试不能代表 RTX 4060 上 batch_size=10 的峰值显存，当前 CUDA 驱动不可用，未报告 GPU 实测值。

推理的 inference_fusion_alpha、forward_one_step、score/batch_score 和 Stage 12–13 均未修改；新模式不增加参数或 checkpoint tensor，不影响已有推理加载接口。

Stage 12–13 同样使用 run_dual.sh，训练完成后手动执行。仅 Valid、alpha=0：

```bash
./run_dual.sh --stage 12 --stop_stage 13 \
  --eval_valid_set false --test_sets org/valid
```

仅 Valid、alpha=0.5（与 alpha=0 使用相同 checkpoint）：

```bash
./run_dual.sh --stage 12 --stop_stage 13 \
  --eval_valid_set false --test_sets org/valid \
  --inference_args "--inference_fusion_alpha 0.5 --num_workers 0" \
  --inference_tag decode_alpha05_fusion
```
