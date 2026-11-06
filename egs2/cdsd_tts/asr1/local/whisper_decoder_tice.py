class OpenAIWhisperDecoder(AbsDecoder, BatchScorerInterface):
    """Transformer-based Speech-to-Text Decoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: Optional[str] = None,
        load_origin_token_embedding=False,
    ):
        try:
            import whisper
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        super().__init__()

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(
            whisper_model, download_root=download_dir, device="cpu"
        )
        self.decoders = copy.deepcopy(_model.decoder)
        attention_dim = self.decoders.token_embedding.embedding_dim

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        # load the original token_embeddings, if the vocabulary is expanded
        self.load_origin_token_embedding = load_origin_token_embedding

        # vocab size mismatch -> reinitialize embedding
        # orig vocab size (multilingual): 51865
        # orig vocab size (english): 51864
        if vocab_size != self.decoders.token_embedding.num_embeddings:
            if self.load_origin_token_embedding:
                assert (
                    vocab_size > self.decoders.token_embedding.num_embeddings
                ), "expanded vocab_size should be larged than the origin"
                self.decoders.token_embedding = ExpandedTokenEmbedding(
                    self.decoders.token_embedding,
                    vocab_size - self.decoders.token_embedding.num_embeddings,
                )
            else:
                orig_emb_std, orig_emb_mean = torch.std_mean(
                    self.decoders.token_embedding.weight
                )
                self.decoders.token_embedding = torch.nn.Embedding(
                    vocab_size, attention_dim
                )
                torch.nn.init.normal_(
                    self.decoders.token_embedding.weight,
                    orig_emb_mean.item(),
                    orig_emb_std.item(),
                )

        self.decoders.train()
        del _model
        # ========================================================添加
        import logging
        from espnet2.asr.encoder.whisper_encoder import FactorizedLoraLinear
        
        for param in self.decoders.parameters():
            param.requires_grad = False

        lora_r = 4 

        modules_to_replace = []
        for name, module in self.decoders.named_modules():
            if isinstance(module, torch.nn.Linear) and module.__class__.__name__ == "Linear":
                if "output_layer" in name or "lm_head" in name:
                    continue
                if any(target in name for target in ["query", "key", "value", "out", "mlp.0", "mlp.2"]):
                    modules_to_replace.append((name, module))

        for name, module in modules_to_replace:
            labels = name.split('.')
            submodule = self.decoders
            for label in labels[:-1]:
                submodule = getattr(submodule, label)
            
            factorized_layer = FactorizedLoraLinear(module, r=lora_r)
            setattr(submodule, labels[-1], factorized_layer)
            
        for name, sub_module in self.decoders.named_modules():
            if sub_module.__class__.__name__ == "FactorizedLoraLinear":
                for param in sub_module.parameters():
                    if id(param) != id(sub_module.original_linear.weight) and \
                       (sub_module.original_linear.bias is None or id(param) != id(sub_module.original_linear.bias)):
                        param.requires_grad = True
            
            # if "ln" in name or "layer_norm" in name:
            #     for param in sub_module.parameters():
            #         param.requires_grad = True
            
        trainable_params = [n for n, p in self.decoders.named_parameters() if p.requires_grad]
        logging.info(f"============ [Decoder Side-Branch Check] ============")
        logging.info(f"Total trainable parameters count in Decoder: {len(trainable_params)}")
        for name in trainable_params[:20]:  # 放开至 20，确保能清晰看到各类参数的层级分布
            logging.info(f" -> Trainable Decoder Part: {name}")

    @torch.no_grad()
    def fuse_factorized_weights(self):
        for name, module in self.decoders.named_modules():
            if module.__class__.__name__ == "FactorizedLoraLinear":
                scaling = 16.0 / 4.0 
                
                delta_W1 = torch.matmul(module.B1, module.A1) * module.a1 * scaling
                delta_W2 = torch.matmul(module.B2, module.A2) * module.a2 * scaling
                delta_W = delta_W1 + delta_W2
                
                module.original_linear.weight.data.add_(delta_W)
                module.a1.data.zero_()
                module.a2.data.zero_()
                module.B1.data.zero_()
                module.B2.data.zero_()
                module.A1.data.zero_()
                module.A2.data.zero_()
        # ========================================================添加结束


#以下是在bin/asr_inference.py文件中修改
class Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    @typechecked
    def __init__(
        self,
        asr_train_config: Union[Path, str, None] = None,
        asr_model_file: Union[Path, str, None] = None,
        transducer_conf: Optional[Dict] = None,
        lm_train_config: Union[Path, str, None] = None,
        lm_file: Union[Path, str, None] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str, None] = None,
        token_type: Optional[str] = None,
        bpemodel: Optional[str] = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        normalize_length: bool = False,
        streaming: bool = False,
        enh_s2t_task: bool = False,
        quantize_asr_model: bool = False,
        quantize_lm: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        hugging_face_decoder: bool = False,
        hugging_face_decoder_conf: Dict[str, Any] = {},
        time_sync: bool = False,
        multi_asr: bool = False,
        lid_prompt: bool = False,
        lang_prompt_token: Optional[str] = None,
        nlp_prompt_token: Optional[str] = None,
        prompt_token_file: Optional[str] = None,
        partial_ar: bool = False,
        threshold_probability: float = 0.99,
        max_seq_len: int = 5,
        max_mask_parallel: int = -1,
    ):

        task = ASRTask if not enh_s2t_task else EnhS2TTask

        if quantize_asr_model or quantize_lm:
            if quantize_dtype == "float16" and torch.__version__ < LooseVersion(
                "1.5.0"
            ):
                raise ValueError(
                    "float16 dtype for dynamic quantization is not supported with "
                    "torch version < 1.5.0. Switch to qint8 dtype instead."
                )

        qconfig_spec = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype: torch.dtype = getattr(torch, quantize_dtype)

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = task.build_model_from_file(
            asr_train_config, asr_model_file, device
        )

        if enh_s2t_task:
            asr_model.inherite_attributes(
                inherite_s2t_attrs=[
                    "ctc",
                    "decoder",
                    "eos",
                    "joint_network",
                    "sos",
                    "token_list",
                    "use_transducer_decoder",
                ]
            )
        # ========================================================添加
        all_params = [n for n, _ in asr_model.named_parameters()]
        logger.info("🔮 ====== [INFERENCE PARAMETER AUDIT START] ======")
        logger.info(f"Total parameter keys inside inference memory: {len(all_params)}")
        
        target_keys = ["A1", "B1", "a1", "a2"]
        for key in target_keys:
            matched = [n for n in all_params if key in n]
            logger.info(f"-> Key '{key}' matched {len(matched)} parameters in inference model.")
            if len(matched) > 0:
                logger.info(f"   Example path: {matched[0]}")
        logger.info("🔮 ====== [INFERENCE PARAMETER AUDIT END] ======")
        if hasattr(asr_model, "encoder") and hasattr(asr_model.encoder, "fuse_factorized_weights"):
            logger.info("⚡ Detected Factorized Side-Branch in Encoder. Fusing weights for zero-latency inference...")
            asr_model.encoder.fuse_factorized_weights()

        if hasattr(asr_model, "decoder") and hasattr(asr_model.decoder, "fuse_factorized_weights"):
            logger.info("⚡ Detected Factorized Side-Branch in Decoder. Fusing weights for zero-latency inference...")
            asr_model.decoder.fuse_factorized_weights()
        # ========================================================添加结束
        asr_model.to(dtype=getattr(torch, dtype)).eval()