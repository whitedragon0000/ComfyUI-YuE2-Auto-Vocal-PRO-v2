"""Prevent the native ABC planner from turning a long *uploaded cover* into 13s.

Unlike a text-to-music prompt, a source-audio cover has real temporal evidence.
Do not replace a long song with a tiny generated ABC when user expects to keep
its musical structure. An unsuitable candidate may fall back only when the
source itself also satisfies the verified score/lyrics contract.
"""
from __future__ import annotations


def audio_seconds(audio):
    if not audio or 'waveform' not in audio or 'sample_rate' not in audio:
        return None
    sample_rate = int(audio['sample_rate'])
    samples = int(audio['waveform'].shape[-1])
    return samples / sample_rate if sample_rate > 0 and samples > 0 else None


def validate_generated_score_duration(generated_seconds, source_audio, max_duration, requested_lyrics=None):
    original = audio_seconds(source_audio)
    if original is None or generated_seconds is None:
        return None  # Unmeasurable duration is handled by strict score parser.
    target = min(float(original), float(max_duration)) if max_duration > 0 else float(original)
    # A drastically shorter ABC is correct for a five-minute source with only
    # a few user-authored lines. The legacy 60% cover threshold is inappropriate
    # in that case. Still reject a collapsed 13s score when the lyrics need
    # substantially more singing time. Never infer permission to add words.
    effective_floor = target * 0.60
    soft_target = None
    if requested_lyrics is not None:
        from .vocal_melody_engine import syllabify
        syllables = len(syllabify(requested_lyrics))
        soft_target = max(32.0, syllables / 2.15 + 30.0)
        if target > soft_target * 1.65:
            effective_floor = min(effective_floor, max(18.0, soft_target * 0.55))
    if target >= 40 and float(generated_seconds) < effective_floor:
        raise ValueError(
            'Official YuE2GenerateABC produced %.1f seconds of musical score from '
            'a %.1f-second uploaded song (minimum conservative cover length %.1fs). '
            'Rejecting drastically shortened arrangement before AR or NAR sampling.'
            % (generated_seconds, original, effective_floor))
    return {'source_audio_seconds': original, 'target_cover_seconds': target,
            'generated_score_seconds': generated_seconds,
            'lyrics_soft_target_s': soft_target, 'minimum_score_seconds': effective_floor}
