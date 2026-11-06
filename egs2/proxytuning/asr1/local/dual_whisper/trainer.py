"""Keep ESPnet updates/checkpoints intact; measure whole-epoch CUDA peaks."""
import logging
import torch
from espnet2.train.trainer import Trainer


class DualTrainer(Trainer):
    @classmethod
    def train_one_epoch(cls, *args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        device = next(model.parameters()).device
        cuda = device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        result = super().train_one_epoch(*args, **kwargs)
        if cuda:
            torch.cuda.synchronize(device)
            logging.info("CUDA training peak memory allocated=%d reserved=%d bytes",
                         torch.cuda.max_memory_allocated(device),
                         torch.cuda.max_memory_reserved(device))
        else:
            logging.info("CUDA training peak memory: unavailable (CPU execution)")
        return result
