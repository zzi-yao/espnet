#!/usr/bin/env python3
"""SpeechLM job template for training configuration."""

from typing import Any, Callable, Dict
import yaml
import re
import torch.nn as nn
import numpy as np

from espnet2.speechlm.template import AbsJobTemplate
from espnet2.speechlm.configuration import SPEECHLM_TASK_CONFIGS
from espnet2.speechlm.multimodal_io import AbsIO, MULTIMODAL_IOS
from espnet2.speechlm.model.speechlm import SPEECHLM_MODELS

from espnet2.speechlm.utils.data import pad_list

class SpeechLMJobTemplate(AbsJobTemplate):
    """Job template for SpeechLM training tasks.

    This class implements the specific model and data processing
    configurations for speech language modeling tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the SpeechLM job template.

        Args:
            config: Dictionary containing job configuration parameters.
        """
        super().__init__(config)

        # (1) keep other configs
        self.config = config
        
        # (2) build tokenizers and vocabulary
        io_config = config['multimodal_io']
        self.multimodal_io = dict()
        for io_name, io_config in io_config.items():
            multimodal_io_class = MULTIMODAL_IOS[io_name]
            assert issubclass(multimodal_io_class, AbsIO)
            self.multimodal_io[io_name] = multimodal_io_class(**io_config)
        
        self.vocab, self.vocab_interval = self._build_vocabulary()

    def _build_vocabulary(self, num_special_tokens=256):
        # (1) Initial special token. We keep a fixed number of slots
        vocab_interval = {
            "special_token": (0, num_special_tokens)
        }
        vocab = [
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|eot|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|text|>",
            "<|audio|>",
            "<|image|>",
            "<|toolcall|>",
        ]
        while len(vocab) < num_special_tokens:
            vocab.append(f"<|unused_{len(vocab)}|>")
        
        # (2) add vocabulary from each discrete multimodal IO.
        start = num_special_tokens
        for io_name, io in self.multimodal_io.items():
            if io.is_discrete:
                vocab.extend(io.get_vocabulary())
                vocab_interval[io_name] = (start, len(vocab))
                start = len(vocab)

        assert len(vocab) == len(set(vocab)), "There are duplicated tokens in the vocab"

        return vocab, vocab_interval
        

    def build_preprocessor(self) -> Callable:
        """Build the data collation function for SpeechLM.

        Returns:
            A callable function for collating SpeechLM batch data.
        """

        processor_config = self.config['preprocessor']
        multimodal_io = {
            io_name: io.copy_for_worker()
            for io_name, io in self.multimodal_io.items()
        }
        return SpeechLMPreprocessor(
            multimodal_io=multimodal_io,
            vocab=self.vocab,
            vocab_interval=self.vocab_interval,
            audio_input=processor_config['audio_input'],
            audio_output=processor_config['audio_output'],
            loss_region=processor_config['loss_region'],
        )

    def build_model(self) -> nn.Module:
        """Build the SpeechLM model.

        Returns:
            A SpeechLM model instance.
        """

        model_config = self.config['model']
        model_class = SPEECHLM_MODELS[model_config['model_choice']]

        model = model_class(
            model_hf_tag=model_config['model_hf_tag'],
            multimodal_io=self.multimodal_io,
            vocab_interval=self.vocab_interval,
            **model_config['model_conf']
        )

        if model_config.get('activation_checkpointing', False):
            model.gradient_checkpointing_enable()
        
        return model


class SpeechLMPreprocessor:
    def __init__(
        self,
        multimodal_io,
        vocab,
        vocab_interval,
        audio_input: str = "continuous_audio",
        audio_output: str = "discrete_audio",
        loss_region: str = "assistant"
    ):
        
        # (1) keep all multimodal_io
        self.multimodal_io = multimodal_io
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.loss_region = loss_region

        # (2) vocabulary
        self.vocab = vocab
        self.vocab_interval = vocab_interval
        self.pad_id = self.vocab.index("<|pad|>")
        self.num_stream = max([
            io.num_stream() 
            for io in multimodal_io.values()
            if io.is_discrete
        ])
    
    def find_length(self, key, data_dict):
        task, _, _ = key
        messages = self._apply_chat_template(task, data_dict)

        # (1) <bos>
        length = 1

        # (2) each message, consider role, modality and end of <eot>/<eos>
        for _, this_io, this_data in messages:
            length += 3
            length += self.multimodal_io[this_io].find_length(this_data)
        
        return length

    def collate_fn(self, data_lst):
        data_lst = [
            self.preprocessing(key, data_dict)
            for key, data_dict in data_lst
        ]
        
        seqs, conti_feats, loss_masks = [], [], []
        for bidx, data_dict in enumerate(data_lst):
            seqs.append(data_dict['sequence'])
            conti_feats.append((bidx, data_dict['conti_feats']))
            loss_masks.append(data_dict['loss_mask'])
        
        print(seqs, 'seqs')
        seqs, _ = pad_list(seqs)
        print(seqs.shape)
        print(loss_masks, 'loss_mask')
        loss_masks, _ = pad_list(loss_masks)
        print(loss_masks.shape)

        conti_feats_dict = dict()
        print('conti_feats: ', conti_feats[0])
        feat = conti_feats[0][1]
        print('length: ', len(feat))
        for f in feat:
            print(f)
        for bidx, (this_io, start, length, feat) in conti_feats:
            if this_io not in conti_feats_dict:
                conti_feats_dict[this_io] = [[], []]
            conti_feats_dict[this_io][0].append((bidx, start, length))
            conti_feats_dict[this_io][1].append(feat)
        
        for io_dict in conti_feats_dict.values():
            io_dict[1], _ = pad_list(io_dict[1])
        
        assert 1 == 2
        return {
            "seqs": seqs,
            "conti_feats": conti_feats_dict,
            "loss_masks": loss_masks
        }
    
    def preprocessing(self, key, data_dict):
        # (1) convert to messages
        task, _, _ = key
        messages = self._apply_chat_template(task, data_dict)

        # (2) initialize
        seq = [self.special_token("<|bos|>")]
        conti_feats = list()
        loss_masks = [self.special_mask(0.0)]
        accum_length = 1

        # (3) loop on each message
        apply_eots = [
            msg1[0] == msg2[0]
            for msg1, msg2 in zip(messages[:1], messages[1:])
        ] + [False]
        for apply_eot, (role, this_io, this_data) in zip(apply_eots, messages):
            apply_loss = float(role == "assistant" or self.loss_region == "all")
            special_mask = self.special_mask(apply_loss)

            # (3.1) role and modality
            seq.append(self.special_token(f"<|{role}|>"))
            loss_masks.append(special_mask)

            modality = self.multimodal_io[this_io].modality
            seq.append(self.special_token(f"<|{modality}|>"))
            loss_masks.append(special_mask)

            accum_length += 2
            
            # (3.2) the exact data processing
            this_seq, conti_feat, loss_mask = self.multimodal_io[this_io].preprocess(this_data)
            assert this_seq.shape == loss_mask.shape

            # (3.3) this_seq
            if self.multimodal_io[this_io].is_discrete:
                modality_bias = self.vocab_interval[this_io][0]
                this_seq = np.where(this_seq == 0, this_seq, this_seq + modality_bias)
            if this_seq.shape[1] < self.num_stream:
                pad_size = self.num_stream - this_seq.shape[1]
                this_seq = np.pad(this_seq, ((0, 0), (0, pad_size)))
            seq.append(this_seq)
            accum_length += this_seq.shape[0]

            # (3.4) conti_feats
            if conti_feat is not None:
                length, feat = conti_feat
                conti_feats.append((this_io, accum_length, length, feat))
            
            # (3.5) loss_mask
            if loss_mask.shape[1] < self.num_stream:
                pad_size = self.num_stream - loss_mask.shape[1]
                loss_mask = np.pad(loss_mask, ((0, 0), (0, pad_size)))
            loss_masks.append(loss_mask * apply_loss)

            # (3.6) <eot> or <eos>
            if apply_eot:
                seq.append(self.special_token(f"<|eot|>"))
            else:
                seq.append(self.special_token(f"<|eos|>"))
            loss_masks.append(special_mask)
            accum_length += 1

        # (4) concat
        seq = np.concatenate(seq, axis=0)
        loss_mask = np.concatenate(loss_masks, axis=0)

        # TODO: Add CFG here
        print('internal conti feats: ', conti_feats)
        data = {
            "sequence": seq,
            "conti_feats": conti_feats,
            "loss_mask": loss_mask,
        }

        # self.dialogue(data)
        return data
    
    def dialogue(self, data):
        seq = data['sequence']
        loss_mask = data['loss_mask']
        conti_feats = data['conti_feats']

        for i, (s, m) in enumerate(zip(seq, loss_mask)):
            s = [self.vocab[s] for s in s.tolist()]
            m = m.tolist()
            print(f"Frame {i} | token: {s} | weight: {m}")
        
        for this_io, conti_start, length, feat in conti_feats:
            print(f"Conti feats: modality={this_io}, conti_feat={conti_start}, length={length}, feat={feat.shape}")

    def special_mask(self, value):
        retval = np.zeros((1, self.num_stream)).astype(np.float32)
        retval[0, 0] = value
        return retval

    def special_token(self, token):
        num_special_token = self.vocab_interval['special_token'][1]
        special_tokens = self.vocab[:num_special_token]
        token_id = special_tokens.index(token)
        retval = np.ones((1, self.num_stream)).astype(np.int64) * self.pad_id
        retval[0, 0] = token_id
        return retval
    
    def _apply_chat_template(self, task, data_dict):
        if "dialogue" in data_dict:
            if len(data_dict) != 1:
                raise ValueError("If dialogue exist, there should be no more other entries")
            return data_dict['dialogue']
        else:
            task_config = SPEECHLM_TASK_CONFIGS[task]
            print('task config: ', task_config)
            messages = list()
            for role, entry in task_config:
                if bool(re.match(r"^audio", entry)):
                    if role == "user" or role == "system":
                        this_io = self.audio_input
                    elif role == "assistant":
                        this_io = self.audio_output
                elif bool(re.match(r"^text", entry)):
                    this_io = "text"
                else:
                    raise ValueError(f"Not supported data entry in template: {entry}")
                
                this_data = data_dict[entry]
                message = (role, this_io, this_data)
                messages.append(message)
            return messages

    def _special_token(self, name):
        token = f"<|{name}|>"
        special_range = self.vocab_interval["special_token"][1]
        return self.vocab[:special_range].index(token)


if __name__ == "__main__":
    config = "test.yaml"
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
        template = SpeechLMJobTemplate(config)
        template.build_model()
