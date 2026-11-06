# 冻结已微调 small，重新训练 tiny

入口 run_dual.sh 默认执行 Stage 11–13，配置为 conf/tuning/train_asr_whisper_dual_smallfft.yaml，输出到新的 exp/asr_whisper_dual_smallfft_tiny。训练目标仍为 fusion_ce_only，small/tiny logits 各占 0.5。没有从原 dual 或 auxce checkpoint 继续训练；新目录首次启动从官方预训练 tiny 初始化。

model_conf.small_model_file 指向：

```
exp/asr_train_asr_whisper_small_finetune_raw_zh_whisper_multilingual_8:1:1_1e-4_small/valid.acc.ave.pth
```

该链接指向 valid.acc.ave_3best.pth。构造模型时先建立相同结构，再严格加载 FFT small 的 encoder 和 decoder 全部参数及 buffer；实际训练中使用的是 FFT 权重，而非初始化时的官方 small 权重。检查模型类型、词表顺序、zh/transcribe 和各 branch 的参数名/维度。FFT small 保持 requires_grad=False、eval、no_grad，optimizer 只包含 tiny。

small 的冻结参数仍不重复保存进 dual checkpoint，因此必须保留此源 FFT 权重和相邻 config.yaml。新 dual 训练的 config.yaml 会保存 small_model_file；恢复训练及 Stage 12–13 构造模型时自动重载同一份 FFT small，然后恢复训练后的 tiny。无需推理时再通过 --small_model_file 覆盖。原模型配置不提供该字段时，继续使用官方 small，旧模型行为保持兼容。

执行：

```bash
source ./path.sh
./run_dual.sh
```

Stage 11 复用原 Stage 1–10 数据，训练完成后 Stage 12–13 使用 valid.loss.ave.pth、推理 alpha=0.5，对 test 与 org/valid 解码评分。解码目录标签 decode_smallfft_alpha05_fusion。后续只做解码可加 --stage 12 --stop_stage 13。新实验目录后续重复启动遵循原 ESPnet resume 行为。

小规模 CPU 检查记录见 exp/dual_whisper_smallfft_checks/results.json：FFT small 全部 tensor 与指定 checkpoint 精确一致；tiny 的全部 tensor 与官方 tiny.pt 精确一致；真实语音单 batch forward/backward 通过；small 无梯度且始终 eval，tiny 梯度有限非零，optimizer 仅包含可训练参数。检查未启动正式训练。
