from .abs_io import AbsIO
from .text import HuggingFaceTextIO
from .audio import DiscreteAudioIO, ContinuousAudioIO

MULTIMODAL_IOS = {
    "text": HuggingFaceTextIO,
    "continuous_audio": ContinuousAudioIO,
    "discrete_audio": DiscreteAudioIO,
}

__all__ = [
    AbsIO,
    MULTIMODAL_IOS,
]