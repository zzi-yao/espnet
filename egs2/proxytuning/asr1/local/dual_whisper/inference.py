"""Standard ESPnet inference with an independent, inference-only logits weight."""
import logging
from espnet2.bin import asr_inference
from .task import DualASRTask
from .model import DualWhisperDecoder

asr_inference.ASRTask = DualASRTask
_OriginalSpeech2Text = asr_inference.Speech2Text


class Speech2Text(_OriginalSpeech2Text):
    # Set only by the standalone CLI, before creating its single recognizer.
    cli_inference_fusion_alpha = None
    cli_small_model_file = None

    def __init__(self, *args, inference_fusion_alpha=None, small_model_file=None, **kwargs):
        alpha = (self.cli_inference_fusion_alpha if inference_fusion_alpha is None
                 else inference_fusion_alpha)
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError('inference_fusion_alpha must be in [0, 1]')
        super().__init__(*args, **kwargs)
        small_file = small_model_file if small_model_file is not None else self.cli_small_model_file
        if small_file is not None:
            self.asr_model.load_small_checkpoint(small_file)
        if alpha is not None:
            if not isinstance(self.asr_model.decoder, DualWhisperDecoder):
                raise TypeError('inference_fusion_alpha requires a dual_whisper checkpoint')
            self.asr_model.decoder.set_inference_fusion_alpha(alpha)
        logging.info('Inference fusion: checkpoint=%s train_alpha=%s inference_alpha=%s',
                     kwargs.get('asr_model_file'), getattr(self.asr_model.decoder, 'alpha', None), alpha)


# Its inherited from_pretrained() resolves Speech2Text in this upstream module.
asr_inference.Speech2Text = Speech2Text


def main(cmd=None):
    parser = asr_inference.get_parser()
    parser.add_argument('--inference_fusion_alpha', type=float, default=None,
                        help='Small logits weight [0,1], inference only; default uses training weight')
    parser.add_argument('--small_model_file', default=None,
                        help='Override frozen small with an ESPnet Whisper-small FFT checkpoint')
    args = vars(parser.parse_args(cmd))
    Speech2Text.cli_inference_fusion_alpha = args.pop('inference_fusion_alpha')
    Speech2Text.cli_small_model_file = args.pop('small_model_file')
    args.pop('config', None)
    asr_inference.inference(**args)


if __name__ == '__main__':
    main()
