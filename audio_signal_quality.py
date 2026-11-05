"""Final-output waveform checks that do NOT modify the Reference Voice engine.

The original reference_quality.py is left bit-for-bit unchanged. The final
SaveAudio audit invokes this independent verifier on the selected waveform.
"""
from __future__ import annotations


def measure_audio_signal_quality(audio):
    from .reference_quality import verify_trimmed_audio
    from math import isfinite
    import torch
    basic = verify_trimmed_audio(audio)
    x = audio['waveform']
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x.detach().float()
    duration = basic['duration_seconds']
    sr = int(audio['sample_rate'])
    n = min(12, max(1, int(duration)))
    size = max(1, min(sr, x.shape[-1]))
    max_start = max(0, x.shape[-1] - size)
    starts = [round(max_start * i / max(1, n-1)) for i in range(n)]
    rms_windows = [float(torch.sqrt(torch.mean(x[..., i:i+size] ** 2))) for i in starts]
    active = sum(rms >= 1e-4 for rms in rms_windows)
    fraction = active / len(rms_windows)
    basic['active_window_fraction'] = round(fraction,4)
    basic['window_rms'] = [round(rms, 6) for rms in rms_windows]
    if duration >= 10 and fraction < 0.25:
        basic['quality_warning'] = 'mostly_silent'
    return basic
