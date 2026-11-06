# 与 Whisper-small FFT 对齐的第二阶段双模型实验

未启动正式训练、全量验证或 Test 解码。参考实验是：

`exp/asr_train_asr_whisper_small_finetune_raw_zh_whisper_multilingual_8:1:1_1e-4_small/`

依据是该目录的 config.yaml、run.sh、train.log、解码日志，以及当前 ESPnet 的 ESPnetASRModel._calc_att_loss、LabelSmoothingLoss、th_accuracy、BeamSearch 和评分脚本。历史公共源码未快照归档，不能证明历史代码逐字节相同；本次记录当前原 ASRModel loss 方法的 SHA256，以使后续实验可追溯。

## 逐项差异与调整

| 项目 | 此前双模型 | 参考 FFT / 本次对齐 | 对可比性的影响 |
| --- | --- | --- | --- |
| 损失实现 | 自定义普通 CE | 直接保留 super() 创建的原 LabelSmoothingLoss | 同一平滑分布与熵常数，不用其他 CE 近似替代 |
| smoothing | 0 | 0.1 | 损失绝对尺度与旧双模型不同，不能直接比较旧 loss |
| decoder 输入 | SOT + 预处理器已给出的提示 + 文本 | 保持相同，直接复用原 add_sos_eos 路径 | 没有新增/重复 zh/task/no-timestamps 前缀 |
| target / EOS | 已有提示位置改为 ignore_id | target=提示+文本+EOS，保留提示监督 | 改变监督位置，恢复参考协议 |
| padding mask | ignore_id，加提示 mask | 仅原 add_sos_eos 产生的 ignore_id padding | 通过原函数委托实现，不是单独删除一行 mask |
| loss 归一化 | sum/B | 原 LabelSmoothingLoss 的 normalize_length=false，除以 B | 不额外再除 B；不是按有效 token 数平均 |
| valid.acc | 排除提示位置的融合 accuracy | 相同 th_accuracy，提示/文本/EOS 都计入，padding 不计 | 定义已测试一致，才设置 acc/max 选模 |
| validation CER | 自定义去提示 TF CER | 原 error_calculator 的全部参考 target 协议，仅日志改名 cer_tf/wer_tf | 两者均是 TF 监控值，不是 Stage 13 真实 CER |
| optimizer | AdamW, lr=1e-4, decay=.01, betas=.9/.99, eps=1e-6 | 相同 | 保持 |
| scheduler | WarmupLR/1500 | 相同 | 保持 |
| clip | L2，1.0 | 相同 | 保持 |
| 训练轮次 | max_epoch=7，可能早停 | max_epoch=7，patience=null | 不因新指标改变而缩短预算 |
| batch / accumulation | folded最大10 / 4 | 相同；fold_length=[80000,150]，排序descending | 名义最大有效 batch=40；folded 下实际不是固定40 |
| 精度/seed | FP32参数+AMP，seed2022 | 相同 | cudnn/TF32 相关开关也复制参考配置 |
| 验证 batch | 1 | null，继承训练最大10，同样 folded | 恢复原 reporter 的 batch 加权聚合方式 |
| workers | 2 | 4 | 对齐数据迭代设置 |
| checkpoint selection | valid.loss/min | valid.acc/max | 原acc定义已核验，选择目标恢复原协议 |
| averaging | loss 标准下3best | acc 标准下3best | 排名机制与参考相同，不保证选中相同 epoch |
| early stopping | patience=3 | null；criterion仍valid.loss/min但不启用早停 | 不因连续无提升而少训 |
| beam / penalties | beam3, penalty0 | 相同；normalize_length=false | 原引擎和结束逻辑不改 |
| 解码长度 | maxlenratio=-128 | 0.25 | 输入encoder帧数×0.25的上限，日志参考样例98→24 |
| 提示/结束条件 | Whisper原tokenizer和primer、EOS | 相同 | 固定primer；EOS结束，达到长度上限原引擎添加EOS；非0 ratio不使用ratio=0的自动end_detect |
| 解码并行 jobs | 1 | 4 | 与原 small FFT 吞吐设置一致，GPU峰值尚待硬件验证 |
| CER / WER 评分 | Stage13 tokenize_text+sclite | 完全复用同一入口，char/word、cleaner/NLS规则相同 | 不用 teacher-forcing CER 替代 |

内部 LabelSmoothingLoss 使用 KLDivLoss 是 ESPnet 对平滑标签的原实现，不是 small/tiny 分布之间的知识蒸馏。本实验只融合 logits 一次，再交给该 criterion；没有新增辅助损失或训练层。

## 监督位置及 accuracy 证据

原预处理器已经给出 `[zh, transcribe, notimestamps, text...]`：

- decoder input：`[SOT, zh, transcribe, notimestamps, text...]`，右侧用 EOS 补齐。
- target：`[zh, transcribe, notimestamps, text..., EOS]`，右侧用 ignore_id 补齐。
- loss：原 LabelSmoothingLoss 对所有非 padding target 监督，分母为 batch size。
- acc：非 padding 位置预测正确数 / 非 padding 位置数，包含三个提示位置和 EOS。
- reporter：仍按 ESPnet 的 batch size 权重聚合；通过对齐验证 batch sampler，使聚合协议一致。不能将它称为全语料 token 数加权的 accuracy。

新 `training_protocol: small_fft` 不复制上述处理逻辑，而是调用原 `ESPnetASRModel._calc_att_loss`，使用本模型的融合 decoder。验证时同样调用。`valid.acc` 的输出定义因此相同，预测 logits 的差异是所比较模型本身的差异。

CPU 两条真实语音（不同长度、含padding）的核验：同一组融合 logits 交给原单模型函数和新路径，decoder input、target、有效位置、loss、acc 全部相同；另构造仅提示位置或EOS位置出错的 logits，确认两路 accuracy 的下降完全相同。small 无梯度，tiny 梯度有限非零。详见 `exp/dual_whisper_smallfft_aligned_checks/results.json`。

重建当前 stats 的 sampler：44,675 条训练语音、每轮4,683个训练batch和583个验证batch，累积4次，每轮1,170个更新边界，7轮8,190个。原 small FFT checkpoint 的 AdamW step 和 scheduler last_epoch 均为8,190。新实验对齐的是更新边界和预算；AMP overflow 可能跳过更新，实际成功更新次数需训练后复查，不能提前保证。每轮末尾3个未满累积窗口的处理沿用原 Trainer。

## 新配置和独立目录

- `conf/tuning/train_asr_whisper_dual_smallfft_aligned.yaml`
- `conf/whisper_decode_dual_smallfft_aligned.yaml`
- `run_dual.sh` 已指向以上配置及 `exp/asr_whisper_dual_smallfft_aligned`。
- small 来源仍为指定 FFT 的 `valid.acc.ave.pth`，冻结；tiny 仍从官方 tiny.pt 初始化，全参数训练；融合0.5/0.5。
- 新 tiny checkpoint：`exp/asr_whisper_dual_smallfft_aligned/valid.acc.ave.pth`，新解码标签 `decode_smallfft_aligned_alpha05`。
- 旧 yaml、旧 checkpoint、旧结果目录均未修改。training_protocol 默认 legacy，旧配置继续使用旧 CE/提示mask。推理 inference_fusion_alpha 保留。

确认配置后才执行：

```bash
source ./path.sh
# 按run_dual.sh配置运行Stage11–13；当前未执行：
./run_dual.sh
# 若希望先只训练，便于先确认Valid再做Test：
./run_dual.sh --stage 11 --stop_stage 11
# 训练后仅Valid：
./run_dual.sh --stage 12 --stop_stage 13 --test_sets org/valid
```

回归检查（不会开始完整训练）：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=local python3 -m dual_whisper.check_aligned
```

## 受控比较的边界

协议可以对齐，但不能说总训练成本或模型容量相同：当前 small 已先做过7轮FFT，双模型第二阶段还会训练 tiny 7轮；small FFT基线只有前一阶段。应报告为“冻结已训练small、第二阶段训练tiny”的实验，单独列出额外成本，而不能把性能差异归因于唯一的结构变化。整个双模型的参数量和推理计算也高于单 small。

当前 CUDA 不可用，未在 RTX4060 8GB 实测对齐配置的 batch10+AMP 峰值显存。Label smoothing 会产生较大的词表维中间tensor，不应把旧纯CE显存数字当作新实测值。没有自动降低batch、累积、epoch或更换optimizer。正式GPU测试若OOM，需先报告预算和采样分组的影响再调整，不能宣称有效batch形式相同就已严格等价。

历史数据和公共源码没有内容哈希快照，因此本次是依据保存配置、日志和当前原 ESPnet 路径建立的可追溯协议对齐，不保证精确复现历史CER数字。尤其不能把原 small 的21.5%当作新双模型必然达到的结果。

## Stage 11 fold_length 追加修复

训练输入 YAML 不再写 fold_length；由 run_dual.sh 的 asr_speech_fold_length=800、asr_text_fold_length=150 经 asr.sh 统一生成 `--fold_length 80000 --fold_length 150`。原 small FFT 的源 YAML 也是如此；其保存的 config.yaml 中 `[80000,150]` 是解析结果，不能在保留命令行 append 参数时直接复制回输入 YAML。

此前失败快照包含 `[80000,150,80000,150]`，但 train_shape_file 只有 speech_shape、text_shape.whisper_multilingual 两项。现在生效值恢复为 `[80000,150]`，顺序音频→文本不变。失败目录中的 config.yaml 保留原快照，下一次通过 run_dual.sh 启动会保存新的解析配置。

回归检查复用了失败日志中的实际 Stage 11 命令，仅将检查运行在 CPU 且 loader_workers=0，以避开沙箱 worker IPC 限制；训练配置仍是 num_workers=4。未构造模型或执行训练，已通过原 build_iter_factory 成功读取 train 和 valid 各一个真实 batch，训练/验证仍为4683/583个batch。报告：`exp/dual_whisper_smallfft_aligned_checks/iterator_check.json`。复查入口：`PYTHONPATH=local python3 -m dual_whisper.check_stage11_iterator`。
