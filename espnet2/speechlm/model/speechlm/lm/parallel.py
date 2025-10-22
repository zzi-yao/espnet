import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig


def ParallelHFModel(model_hf_tag, **kwargs):
    model_class = build_parallel_hf_class(model_hf_tag)
    return model_class.from_pretrained(model_hf_tag, **kwargs)


def build_parallel_hf_class(model_hf_tag):

    config = AutoConfig.from_pretrained(model_hf_tag)
    architecture = config.architectures[0]
    architecture = getattr(transformers, architecture)

    class ParallelLLM(architecture):
        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            multimodal_io,
            vocab_interval,
            **kwargs,
        ):
            # (1) Load the base model using parent's from_pretrained
            model = super(ParallelLLM, cls).from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )

            # (2) rebuild the input/output embedding tables. 
            # (a) place 0 as all-zero embedding
            # (b) replace text embeddings from the pre-trained weights.
            with torch.no_grad():
                vocab_size = max([end for _, end in vocab_interval.values()])

                embed_dim = model.config.hidden_size
                new_embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                new_lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
                new_embed_tokens.weight[0] = 0.0
                new_lm_head.weight[0] = 0.0

                if 'text' in vocab_interval:
                    
                    if not (hasattr(model, "model") and hasattr(model.model, "embed_tokens")):
                        raise AttributeError("Model must have 'model.embed_tokens' attribute")
                    if not hasattr(model, "lm_head"):
                        raise AttributeError("Model must have 'lm_head' attribute")
                    
                    text_start, text_end = vocab_interval['text']
                    
                    old_embed = model.model.embed_tokens
                    old_lm_head = model.lm_head
                    orig_vocab_size = old_embed.weight.shape[0]

                    if text_end - text_start != orig_vocab_size:
                        raise ValueError(
                            f"text_end - text_start ({text_end - text_start}) must equal "
                            f"original vocab size ({orig_vocab_size})"
                        )

                    embed_dim = model.config.hidden_size
                    new_embed_tokens = nn.Embedding(vocab_size, embed_dim)
                    new_lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
            
                    new_embed_tokens.weight[text_start:text_end] = old_embed.weight
                    new_lm_head.weight[text_start:text_end] = old_lm_head.weight

                model.model.embed_tokens = new_embed_tokens
                model.lm_head = new_lm_head

            # (3) build stream embeddings. First stream doesn't apply this embedding
            possible_num_stream = [io.num_stream() for io in multimodal_io.values() if io.is_discrete]
            if len(possible_num_stream) == 0:
                raise ValueError("Cannot proceed with all IOs are continuous")
            model.num_stream = max(possible_num_stream)
            model.stream_emb = nn.Embedding(model.num_stream, embed_dim)

            # (4) multimodal IO
            model.vocab_interval = vocab_interval
            model.multimodal_io_dict = nn.ModuleDict(multimodal_io)
            model.adaptor = nn.ModuleDict()
            for io_name, io in model.multimodal_io_dict.items():
                if not io.is_discrete:
                    model.adaptor[io_name] = nn.Linear(
                        io.feature_dim(), model.config.hidden_size,
                    )

            return model

        def encode(
            self, input_ids, position_ids, use_cache=False, past_key_value=None
        ):
            assert input_ids.dim() == 3

            inputs_embeds = self.model.embed_tokens(input_ids).sum(dim=2)

            # TODO(Jinchuan) add continuous feature
            output = super(ParallelLLM, self).forward(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )

            stream_emb = self.stream_emb.weight
            stream_emb[0] = 0.0
            output = output.unsqueeze(2) + stream_emb.tile(1, 1, 1, 1)

            return output.last_hidden_state, output.past_key_value
        
        def forward(
            self, input_ids, position_ids, use_cache=False, past_key_value=None
        ):
            
            last_hidden_state, _ = self.encode(
                input_ids, position_ids, use_cache, past_key_value
            )

            loss, stats = self._compute_loss(last_hidden_state, input_ids)

            return loss, stats
        
        def _compute_loss(self, hidden_states, input_ids):
            raise NotImplementedError
        
        def multimodal_forward(self, hidden_stats, conti_feats):
            raise NotImplementedError
        
        def register_multimodal_io(self, multimodal_io_dict):
            self.multimodal_io_dict = nn.ModuleDict(multimodal_io_dict)
            self.adaptor = nn.ModuleDict()
            for io_name, io in self.multimodal_io_dict.items():
                if not io.is_discrete:
                    self.adaptor[io_name] = nn.Linear(
                        io.feature_dim(), self.config.hidden_size,
                    )

        def inference(self):
            raise NotImplementedError

    return ParallelLLM
