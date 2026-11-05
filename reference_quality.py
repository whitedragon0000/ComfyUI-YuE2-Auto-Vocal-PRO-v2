"""Reference Voice output-duration and acoustic sanity contracts.

max_duration in stock YuE2 is an upper bound, not a promise of exact length.
A reference prefix can consume most of the samples if semantic generation
ends early. Quality checks fail closed before exposing broken WAVs.
"""
from __future__ import annotations
import json
import math


def nominal_target_seconds(plan_report) -> float | None:
    try:
        data = json.loads(plan_report) if isinstance(plan_report, str) else dict(plan_report)
        value = float(data.get('target_nominal_seconds', 0))
        return value if math.isfinite(value) and value > 0 else None
    except (ValueError, TypeError, AttributeError):
        return None


def minimum_target_seconds(score_seconds, budget_seconds) -> float:
    """Conservative minimum for a *planned* song, never an exact-duration claim.

    Reject grossly premature termination (e.g. ~13 seconds for a 190-second
    source arrangement). A short scored musical excerpt remains short even if
    the upper bound is large.
    """
    if score_seconds is None:
        return 8.0
    score = min(float(score_seconds), float(budget_seconds))
    if score < 20:
        return max(3.0, score * 0.50)
    return max(8.0, score * 0.72)


def validate_semantic_duration(total_frames, prefix_frames, min_target_seconds):
    if not isinstance(total_frames, int) or total_frames <= prefix_frames:
        raise RuntimeError('YuE2 semantic generation produced no cover after the vocal reference prefix.')
    output_s = (total_frames - prefix_frames) / 25.0
    if output_s + 0.041 < float(min_target_seconds):
        raise RuntimeError('YuE2 prematurely ended the song: %.2fs after voice reference; '
                           'minimum expected from the ABC arrangement is %.2fs. '
                           'The result is rejected before acoustic sampling, not exported as a short/noisy cover.'
                           % (output_s, min_target_seconds))
    return output_s


def verify_trimmed_audio(audio, *, minimum_seconds=0.0):
    import torch
    samples = audio['waveform']
    sr = int(audio['sample_rate'])
    if sr <= 0 or samples.ndim != 3 or samples.shape[-1] <= 0:
        raise RuntimeError('Reference Voice produced an empty audio tensor')
    seconds = samples.shape[-1] / sr
    if minimum_seconds > 0 and seconds + 0.06 < minimum_seconds:
        raise RuntimeError('Reference Voice output has only %.2f seconds after trimming the '
                           'voice prefix; expected at least %.2f seconds from its ABC score.'
                           % (seconds, minimum_seconds))
    # Check gross corruption only. Do not reject quiet intros or legitimate drums.
    x = samples.detach().float() if isinstance(samples, torch.Tensor) else torch.as_tensor(samples, dtype=torch.float32)
    if not bool(torch.isfinite(x).all()):
        raise RuntimeError('Reference Voice decoded NaN/Inf samples; audio rejected')
    peak = float(x.abs().max())
    rms = float(torch.sqrt(torch.mean(x*x)))
    if peak < 1e-5 or rms < 1e-6:
        raise RuntimeError('Reference Voice audio is effectively silent; not exporting a broken track')
    # Diagnostic only: music can legitimately contain sharp transients and noise.
    # Do not mislabel percussion as a corrupted vocal track.
    finite_clip_fraction = float((x.abs() >= 0.999).float().mean())
    return {'duration_seconds': round(seconds, 3), 'rms': round(rms, 5),
            'peak': round(peak, 5), 'near_clipping_fraction': round(finite_clip_fraction, 5),
            'quality_warning': ('high_saturation' if finite_clip_fraction > 0.10 else None),
            'scope': 'gross signal corruption only; listening is needed to detect crackling'}
