"""Assert that ESPnet saved/resumed tiny, AdamW, and WarmupLR consistently."""
import json
import os
from pathlib import Path
import editdistance
import torch


def main():
    root = Path(os.environ.get('DUAL_SMOKE_DIR', 'exp/dual_whisper_smoke'))
    p = root / 'model'
    a = torch.load(p / '1epoch.pth', map_location='cpu', weights_only=False)
    b = torch.load(p / '2epoch.pth', map_location='cpu', weights_only=False)
    c = torch.load(p / 'checkpoint.pth', map_location='cpu', weights_only=False)
    assert all(not k.startswith(('encoder.small.', 'decoder.small.')) for k in c['model'])
    assert set(c['model']) == set(b)
    assert all(torch.equal(v, b[k]) for k, v in c['model'].items())
    steps = {int(v['step']) for v in c['optimizers'][0]['state'].values()}
    assert steps == {2}, steps
    assert c['schedulers'][0]['last_epoch'] == 2, c['schedulers']
    for part in ('encoder', 'decoder'):
        assert any(not torch.equal(a[k], b[k]) for k in a if k.startswith(f'{part}.tiny.'))
    assert 'training was resumed' in (root / 'resume.log').read_text()
    for name in ('train.log', 'resume.log'):
        log = (root / name).read_text()
        assert 'beam_search:' not in log, name
        assert 'cer_tf=' in log and 'wer_tf=' in log, name
    assert (p / 'valid.loss.ave.pth').exists()
    config = (p / 'config.yaml').read_text()
    assert 'small_model_name: small' in config and 'tiny_model_name: tiny' in config
    def read_text(path):
        return {fields[0]: fields[1] if len(fields) > 1 else ''
                for line in path.read_text().splitlines()
                if (fields := line.split(maxsplit=1))}
    refs = read_text(root / 'valid.text')
    hyps = read_text(root / 'decode/1best_recog/text')
    assert set(refs) == set(hyps)
    errors = total = 0
    for key in refs:
        ref, hyp = refs[key].replace(' ', ''), hyps[key].replace(' ', '')
        errors += editdistance.eval(ref, hyp)
        total += len(ref)
    report = dict(model_keys=len(b), frozen_small_keys=0, optimizer_steps=sorted(steps),
        scheduler_last_epoch=c['schedulers'][0]['last_epoch'],
        weights_bytes=(p / '2epoch.pth').stat().st_size,
        checkpoint_bytes=(p / 'checkpoint.pth').stat().st_size,
        tiny_changed_after_resume=True, autoregressive_smoke_cer=errors / max(total, 1),
        char_errors=errors, reference_chars=total,
        note='Smoke test uses 8-token decoding cap, not an ASR quality evaluation.')
    (root / 'checkpoint_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
