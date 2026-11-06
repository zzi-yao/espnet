# FFT-small 双模型审计

本次只检查，不启动训练/解码，也不静默改变损失协议或选模配置。

## 权重及优化器

- 源：`exp/asr_train_asr_whisper_small_finetune_raw_zh_whisper_multilingual_8:1:1_1e-4_small/valid.acc.ave.pth` → `valid.acc.ave_3best.pth`。
- 实际构造模型并捕获 load_state_dict 返回值：encoder 187 个 tensor，decoder 292 个 tensor；两次 strict=True，missing_keys=[]，unexpected_keys=[]。
- 源权重没有 encoder/decoder 之外的键。两路 small 的全部 tensor 与源文件逐项完全相等，不存在通过分支过滤而忽略的额外 checkpoint 键。
- small 240,582,912 参数，可训练 0；外部 train() 后仍 eval。
- tiny 的所有 tensor 与 `~/.cache/whisper/tiny.pt` 官方初始化完全相等；37,184,640 参数全部可训练。
- optimizer 恰好包含 166 个 tiny 参数 tensor，与全部 requires_grad=True 参数的对象 ID 集合完全相同；没有 small 参数，没有旧 optimizer state。
- 双模型 state_dict 不含 small，严格恢复检查通过。新配置保存 small_model_file，恢复训练/推理依赖源 FFT 文件，并非重新采用官方 small。
- 完整结果、SHA256 和运行时配置：`exp/dual_whisper_smallfft_checks/audit.json`。

## 路径

新目录为 `exp/asr_whisper_dual_smallfft_tiny`，检查时尚不存在；训练将保存该目录的 config.yaml、checkpoint.pth 和 epoch weights，验证选模产生 valid.loss.ave.pth，Stage 12 读取同一目录的 config.yaml 和 valid.loss.ave.pth。init_param=[]，不从旧 dual/auxce 实验读取 tiny 或 optimizer。small 源文件是按要求唯一引用的旧 FFT 参数。

以后重复启动时 ESPnet 默认 resume 只读取本新目录的 checkpoint.pth。若人为复制旧 checkpoint 进入新目录或通过命令行覆盖 asr_exp/init_param，本检查的默认路径保证就不再适用。

## 21.5% 结果关联

源 FFT 目录的 RESULTS.md 和 score_cer/result.txt 记录 Test CER 21.5%、Valid CER 21.8%，其训练日志记录按 valid.acc 平均 3 个模型。解码日志指定 valid.acc.ave.pth。历史日志保留的是重命名前的通用实验目录名，不带当前后缀；因此可以关联到该实验保存的结果，但没有当时的权重 SHA256，不能独立证明历史运行时文件与今天文件逐字节相同。当前文件哈希已记录在 audit.json。

## 与单独 small FFT 的对比

| 项目 | 单独 small FFT 实际配置/日志 | 当前新双模型 | 结论 |
| --- | --- | --- | --- |
| optimizer | AdamW, lr=1e-4, decay=.01, betas=.9/.99, eps=1e-6 | 相同 | 一致 |
| scheduler | WarmupLR, warmup=1500 | 相同 | 一致 |
| epoch / train batch / accumulation | 7 / folded最大10 / 4 | 相同 | 一致 |
| clipping / precision / seed | norm2=1.0 / FP32参数+AMP / 2022 | 相同 | 一致 |
| train/valid、stats 路径 | dump/raw/train、valid；同名 stats | 复用同一路径 | 配置一致；历史数据无哈希快照 |
| tokenizer / language / task | whisper_multilingual / zh / transcribe | 相同 | 一致 |
| CTC / length normalization | 0 / false | 相同 | 一致 |
| label smoothing | 0.1 | 0.0（纯 CE） | **不一致，改变它会改变已要求的损失协议** |
| prompt token 监督 | 当前原 ESPnet ASRModel 不显式屏蔽 | 双模型屏蔽三个提示位置 | **不一致；历史公共源码版本未归档，实际历史 masking 无法完全追溯** |
| 验证 | teacher-forcing, batch最大10 | teacher-forcing, batch=1，cer_tf/wer_tf明确命名且去提示 | **同为 TF，但统计协议不同** |
| best / averaging | valid.acc/max，3best | valid.loss/min，3best | **排名标准不同，平均数量相同** |
| patience | null | 3 | **不一致，当前可能提前结束** |
| workers | 4 | 2 | 不一致 |
| beam / CTC / LM / minlen / penalty | 3 / 0 / 0 / 0 / 0 | 相同 | 一致 |
| 解码 maxlenratio | 0.25，日志例：输入98→上限24 | -128，固定上限128 | **不一致** |
| 并行解码 job | 4 | 1 | 吞吐设置不同 |

不能据此声称“其他流程全部与原 small FFT 一致”。若要求严格对齐，需要选择是否也把 smoothing、提示监督、验证、选模、早停及解码长度一起改变；其中 smoothing/masking 与之前要求保持纯融合 CE 的约定存在冲突，须先明确目标。

复查命令：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=local python3 -m dual_whisper.check_smallfft
```
