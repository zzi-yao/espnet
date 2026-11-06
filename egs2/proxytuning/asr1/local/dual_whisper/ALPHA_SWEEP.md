# 推理融合系数消融（不训练）

训练系数仍是 model_conf.fusion_alpha=0.5。新增 CLI/API 参数 inference_fusion_alpha，仅覆盖 decoder.forward_one_step 的融合系数；teacher-forcing forward 不读取此参数。默认 None 与原逻辑相同，支持闭区间 [0,1]。

```bash
source ./path.sh
./run_alpha_sweep.sh --preflight-only
./run_alpha_sweep.sh
```

默认固定原实验 `exp/asr_whisper_dual_small_tiny/valid.loss.ave.pth` 的实际目标文件 `valid.loss.ave_3best.pth`。不使用独立 FFT tiny。默认数据为 `dump/raw/org/valid`，与原 Dev CER 30.6% 一致，不把 `test` 当 Dev。检查 training alpha=0.5、model=dual_whisper。manifest 保存 checkpoint、训练配置、small.pt、Dev scp/text/utt2spk 的 SHA256，以及解码配置和设备。所有 alpha 使用同一个已解析的 checkpoint 路径。源配置和 checkpoint 不写入。

扫描值为 0、0.1、0.2、0.3、0.4、0.5、0.7、1。先跑 0.5，以检查原基线，再跑其余值。全部复用 asr.sh stages 12–13（skip_train=true），beam=3，maxlenratio=-128，原 Whisper tokenizer/prompt，原 ESPnet tokenize_text + sclite 的中文 CER/WER 协议。每个输出目录只放源配置/权重的符号链接和独立解码配置、日志、文本、评分结果。既有解码文本不覆盖；已完成 alpha 可复用，不完整目录会报错，需自行将其移至其他目录后重试。

默认输出：

- `exp/dual_whisper_alpha_sweep/manifest.json`：固定输入的审计信息。
- `preflight.json`：真实 Dev 小样本 0/.5/1 的逐步 logits、完整识别、权重校验。
- `pipeline_preflight.json`：小样本通过实际 CLI 和 stages 12–13 的检查。
- `dev/alpha_0.0/` 等：各 alpha 的独立输出；实际文本和评分在 `decode/org/valid/`。
- `dev_results.csv`：按 Dev CER 升序排列；扫描过程中只包含已完成项。
- `best_alpha.json`：当前最优，`complete=true` 才是完整扫描后的选择。
- `status.json`：当前阶段、alpha、进程 PID、输出目录。

CSV 字段：alpha、small_weight、tiny_weight、dev_cer_percent、dev_wer_percent、decode_seconds、decode_result_dir。解码耗时取 run.pl 的 decoder job elapsed time，包含模型加载，不含 stage 13 评分；另外 timing.json 保存 stages 总耗时。

小规模检查：

1. tiny checkpoint 的全部 state_dict tensor 与加载后的 tiny 逐项相同，small 与 small.pt 的 encoder/decoder state_dict 逐项相同。
2. 同一个真实语音、同一 token 前缀，经实际 batch_score → forward_one_step 路径捕获两路 logits。0 权重的 fused logits 等于 tiny，1 权重等于 small，0.5 的 log probabilities 与未覆盖的原路径逐位相同；两端 logits 必须不同。
3. 在 no_grad 下对 0/.5/1 都做真实自回归解码，所有参数 grad=None，全部参数和 buffer 的哈希前后一致。
4. 使用实际 CLI --inference_fusion_alpha 0.5 和原评分 stages 做子集端到端验证。
5. 完整 Dev 的 0.5 如果与 30.6% 相差超过 0.5 个百分点，停止后续扫描，不选择新的权重。CPU/GPU 数值和平台差异可能导致小幅变化。

可指定其他已完成双模型实验或新的输出目录：

```bash
./run_alpha_sweep.sh --experiment exp/asr_whisper_dual_small_tiny \
  --checkpoint valid.loss.ave.pth --output exp/dual_whisper_alpha_sweep_gpu --device cuda
```

同一输出目录固定设备；更换设备或任一输入必须选择新目录。CUDA 不可用时默认自动使用 CPU。完整 5,585 条 Dev 的八组自回归扫描可能持续很久，检查当前 alpha 的 `stages.log` 和 `decode/org/valid/logdir/asr_inference.1.log` 获取进度。不要同时对同一输出目录启动两个进程。

也可以单独通过标准入口设置参数（不修改模型训练配置）：

```bash
./local/dual_python.sh -m espnet2.bin.asr_inference \
  --asr_train_config exp/asr_whisper_dual_small_tiny/config.yaml \
  --asr_model_file exp/asr_whisper_dual_small_tiny/valid.loss.ave.pth \
  --config conf/whisper_decode_dual.yaml --inference_fusion_alpha 0.2 \
  --data_path_and_name_and_type dump/raw/org/valid/wav.scp,speech,kaldi_ark \
  --num_workers 0 --ngpu 1 --output_dir exp/alpha02_manual
```

## 手动确认后才运行 Test

扫描本身不会解码 Test。Dev 扫描全部完成、检查 best_alpha.json 后，可手动执行：

```bash
./run_alpha_sweep.sh --test-best
# 或手动指定一个已确认的 alpha：
./run_alpha_sweep.sh --test-alpha 0.2
```

这两个显式命令仅解码一次 Test，不扫描 Test，不用 Test 选择 alpha。输出在 `test/alpha_X.X/`，结果在 test_result.json，Dev CSV 不改动。

## 结果解释

- best alpha>0 且优于 alpha=0：冻结 small 在推理融合时提供正向信息。
- best alpha=0：在所扫描权重和当前解码条件下，加入 small 未带来改善。
- 联合训练 tiny 的 alpha=0 应与独立 FFT tiny 的 Dev CER 29.1% 比较；若超过 29.1%，表示这次联合训练得到的 tiny 未达到该基线。
- 即使融合最优结果优于 29.1%，也不能仅凭此次推理扫描确定“训练期间冻结 small”具有因果收益；还需要控制训练条件的对照实验。此次不开展额外训练。
- 未完成的 CSV/最佳值只代表已完成组，不能当作最终消融结论。

完整扫描后还会生成 `dev_analysis.md`：包含八组比较表、最优 alpha、相对独立 tiny FFT Dev 29.1% 的百分点差异，以及相对本次联合训练 tiny 单独解码（alpha=0）的比较。百分比使用原 sclite 汇总精度；同 CER 时优先选择较小 alpha。
