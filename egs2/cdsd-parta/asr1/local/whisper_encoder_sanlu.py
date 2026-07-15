#以下是修改后的代码
class OpenAIWhisperEncoder(AbsEncoder):
    """Transformer-based Speech Encoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    @typechecked
    def __init__(
        self,
        input_size: int = 1,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: Optional[str] = None,
        use_specaug: bool = False,
        specaug_conf: Union[dict, None] = None,
        do_pad_trim: bool = False,
    ):
        try:
            import whisper
            from whisper.audio import HOP_LENGTH, N_MELS, N_SAMPLES
            # 引入 torchaudio 用来安全动态生成不同窗长的 Mel 滤波器组
            import torchaudio.functional as F_audio
        except Exception as e:
            print("Error: whisper is not properly installed.")
            raise e

        super().__init__()

        # ==================== 【核心修改 1：多尺度核心参数规范化】 ====================
        self.hop_length = HOP_LENGTH   # 160 (10ms) -> 确保三路特征都是 100Hz，天然时间步对齐
        self.n_fft = 1024              # 放大 FFT 窗口，容纳 50ms 窗长（800点）
        self.n_mels = N_MELS           # 80
        
        # 三窗口：10ms (160点), 25ms (400点), 50ms (800点)
        self.win_lengths = [160, 400, 800] 

        # 【避坑修改】：动态生成适配 n_fft=1024 的通用 Mel 滤波器组，避免原厂 400 点矩阵冲突
        # 使用 torch.nn.Parameter 确保它能随模型一起 move 到 GPU
        import librosa
        mel_fb = librosa.filters.mel(sr=16000, n_fft=self.n_fft, n_mels=self.n_mels)
        self.register_buffer("custom_mel_filters", torch.from_numpy(mel_fb).float())

        self.dropout = torch.nn.Dropout(dropout_rate)

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(
            whisper_model, download_root=download_dir, device="cpu"
        )
        self.encoders = copy.deepcopy(_model.encoder)
        self.encoders.train()
        del _model

        if use_specaug:
            self.specaug = SpecAug(**specaug_conf)
        else:
            self.specaug = None

        self.do_pad_trim = do_pad_trim
        self.pad_samples = N_SAMPLES

        # ==================== 【核心修改 2：多尺度注意力决策网络】 ====================
        # 输入是三路特征拼接（80 * 3 = 240），输出 3 个标量 Scales 作用于三个通道
        self.lora_attention_fc = nn.Sequential(
            nn.Linear(self.n_mels * 3, self.n_mels),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # 增加轻量 dropout 增强鲁棒性
            nn.Linear(self.n_mels, 3)
        )
        
        # 显式设定 requires_grad=True（其实 nn.Sequential 默认就是 True，这样写更稳妥）
        for param in self.lora_attention_fc.parameters():
            param.requires_grad = True



    def output_size(self) -> int:
        return self.encoders.ln_post.normalized_shape[-1]

    def pad_or_trim(
        self,
        array: torch.Tensor,
        length: int,
        axis: int = -1,
    ) -> torch.Tensor:
        """Pad or trim the audio array to N_SAMPLES.

        Used in zero-shot inference cases.
        """
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length).to(array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

        return array

    # ==================== 【核心修改 3：重构多尺度特征提取与融合】 ====================
    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Use log-mel spectrogram computation native to Whisper training"""
        
        mel_scales = []
        
        # 1. 获取老版本 Whisper 固定的 201 维滤波器组 (形状: 80, 201)
        filters = self.custom_mel_filters
        
        # 2. 循环提取三个不同窗长的声学特征
        for win_len in self.win_lengths:
            # 【安全补丁 1】：直接在创建时锁定设备，杜绝跨设备阻塞
            window = torch.hann_window(win_len, device=audio.device)
            
            # --- 物理约束：50ms 必须用 1024点 FFT ---
            if win_len > 400:
                current_n_fft = 1024  
            else:
                current_n_fft = 400   

            stft = torch.stft(
                audio, 
                n_fft=current_n_fft, 
                hop_length=self.hop_length, 
                win_length=win_len,        
                window=window, 
                center=True,
                return_complex=True
            )

            # whisper 默认去掉最后一帧
            magnitudes = stft[..., :-1].abs() ** 2 

            # --- 维度无缝对齐 ---
            if magnitudes.shape[1] != filters.shape[-1]:
                # 【安全补丁 2】：强转为 float32 规避 AMP 混合精度插值报错
                orig_dtype = magnitudes.dtype
                magnitudes = magnitudes.to(torch.float32).permute(0, 2, 1) 
                
                magnitudes = F.interpolate(
                    magnitudes, 
                    size=filters.shape[-1], 
                    mode='linear', 
                    align_corners=False
                )
                magnitudes = magnitudes.permute(0, 2, 1).to(orig_dtype)

            # 矩阵乘法得到麦尔谱
            mel_spec = filters @ magnitudes 

            log_spec = torch.clamp(mel_spec, min=1e-10).log10()

            # 保持 Whisper 原生的动态范围限制与归一化
            log_spec = torch.maximum(
                log_spec,
                log_spec.view(audio.size(0), -1).max(dim=-1)[0][:, None, None] - 8.0,
            )
            log_spec = (log_spec + 4.0) / 4.0
            
            mel_scales.append(log_spec)

        # ---- 截断对齐时间轴 ----
        min_t = min([m.shape[-1] for m in mel_scales])
        mel_scales = [m[..., :min_t] for m in mel_scales]

        # 估算输出特征的有效帧长
        if ilens is not None:
            olens = ilens // self.hop_length
            olens = torch.clamp(olens, max=min_t)
        else:
            olens = torch.full((audio.size(0),), min_t, dtype=torch.long, device=audio.device)

        # ---- 尺度注意力机制 (Scale Attention) ----
        # 【算法补丁 3】：排除有效长度之外的 0 Padding 污染，精确提取全局统计量
        s1_list, s2_list, s3_list = [], [], []
        for i in range(audio.size(0)):
            valid_len = olens[i].item()
            s1_list.append(mel_scales[0][i, :, :valid_len].mean(dim=-1, keepdim=True))
            s2_list.append(mel_scales[1][i, :, :valid_len].mean(dim=-1, keepdim=True))
            s3_list.append(mel_scales[2][i, :, :valid_len].mean(dim=-1, keepdim=True))
        
        s1 = torch.stack(s1_list, dim=0).squeeze(-1) # (B, 80)
        s2 = torch.stack(s2_list, dim=0).squeeze(-1) # (B, 80)
        s3 = torch.stack(s3_list, dim=0).squeeze(-1) # (B, 80)
        
        # 拼接跨尺度的统计信息
        S_g = torch.cat([s1, s2, s3], dim=-1) # (B, 240)
        S_g = S_g.to(device=audio.device, dtype=audio.dtype)
        
        # 计算自适应决策权重 alpha
        alpha = self.lora_attention_fc(S_g) 
        alpha = F.softmax(alpha, dim=-1) 

        # 动态自适应加权融合特征
        enhanced_mel = (
            alpha[:, 0].view(-1, 1, 1) * mel_scales[0] +
            alpha[:, 1].view(-1, 1, 1) * mel_scales[1] +
            alpha[:, 2].view(-1, 1, 1) * mel_scales[2]
        )
        
        if self.training and torch.rand(1) < 0.01: 
            print(f"\n[Scale Weights Check] alpha[0]: {alpha[0].detach().cpu().numpy()}")

        # 如果最层外不需要 olens，返回时可以适配原版包装
        if ilens is None:
            return enhanced_mel
            
        return enhanced_mel, olens
    
    def whisper_encode(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        # 1. 骨干网络的 CNN 前端 (100Hz -> 25Hz)
        x = F.gelu(self.encoders.conv1(input))
        x = F.gelu(self.encoders.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = self.encoders.positional_embedding.size(0)
        
        # 2. 位置编码注入
        if n_frames <= max_pos:
            x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            # 由于位置编码硬编码限制，超过 30 秒的帧（3000帧）将被截断
            x = x[:, :max_pos, :] + self.encoders.positional_embedding

        x = self.dropout(x)

        # 3. 经过 Transformer 编码器块
        for layer, block in enumerate(self.encoders.blocks):
            x = block(x)
            if layer < len(self.encoders.blocks) - 1:
                x = self.dropout(x)

        x = self.encoders.ln_post(x)

        # ==================== 【核心修改：修正时序长度折算】 ====================
        if ilens is not None:
            # 因为在 log_mel_spectrogram 里传出的 ilens 已经是 100Hz 级别的谱帧长
            # 经过 conv1 (stride=2) 和 conv2 (stride=2) 后，整体缩减了 4 倍
            # 使用标准的向下取整除法（// 4）最稳健，且可以完美适配卷积边界
            olens = ilens // 4
            
            # 必须受到当前 Transformer 实际输出的最大物理帧数限制
            olens = torch.clamp(olens, max=n_frames)
        else:
            olens = None

        return x, olens

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.do_pad_trim:
            xs_pad = self.pad_or_trim(xs_pad, self.pad_samples)

        # 此时得到的 feats 已经是加权融合后的多尺度病理特征增强谱了
        feats, feats_lens = self.log_mel_spectrogram(xs_pad, ilens)

        if self.specaug is not None and self.encoders.training:
            feats = torch.transpose(feats, 1, 2)
            feats, feats_lens = self.specaug(feats, feats_lens)
            feats = torch.transpose(feats, 1, 2)

        xs_pad, olens = self.whisper_encode(feats, feats_lens)

        return xs_pad, olens, None

