"""Bound native Standard YuE2 semantic generation to the PLANNED score length.

max_duration remains a user ceiling, not an exact length. Using the user
ceiling directly for a much shorter score lets the AR decoder fill the extra
seconds with uncontrolled singing after the last authorized lyric section.
"""
from __future__ import annotations
import json
import math

from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc


def generation_budget(abc: str, *, max_duration: float = 0.0,
                      expansion_allowance: float = 0.08,
                      musical_tail_seconds: float = 4.0) -> tuple[int,dict]:
    if not math.isfinite(float(max_duration)) or float(max_duration) < 0:
        raise ValueError('max_duration must be finite and nonnegative')
    score = parse_abc(abc)
    actual = float(score.voices['Vocal'].time * 60 / score.bpm)
    if actual <= 0:
        raise ValueError('Planned score has no musical duration')
    desired = actual * (1 + expansion_allowance) + musical_tail_seconds
    ceiling = float(max_duration) if max_duration > 0 else min(900.0, desired)
    # The native generator expects an integer budget. Floor the user ceiling,
    # never round above it. If planning exceeds this ceiling, fail before AR.
    budget = min(int(math.floor(ceiling)), int(math.ceil(desired)))
    if budget < 1:
        raise ValueError('Maximum duration is below native one-second budget')
    if actual > budget + 0.01:
        raise ValueError(f'Planned score {actual:.2f}s exceeds the {budget}s semantic budget; '
                         'the ABC must be shortened or the duration limit increased')
    return budget, {
        'duration_contract': 'planned_score_bounded_semantic_generation',
        'planned_score_seconds': round(actual, 3),
        'user_maximum_seconds': float(max_duration),
        'semantic_token_budget_seconds': budget,
        'expansion_allowance': expansion_allowance,
        'musical_tail_allowance_s': musical_tail_seconds,
        'budget_is_target_length': False,
        'notice': 'Native YuE2 can still end earlier or emit unwanted words; audio requires ASR/listening.'}


class YuE2AdaptiveDurationBudget:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'abc': ('STRING', {'forceInput': True}),
            'max_duration': ('FLOAT', {'forceInput': True}),
        }}
    RETURN_TYPES = ('FLOAT','STRING')
    RETURN_NAMES = ('semantic_max_duration','duration_report')
    FUNCTION = 'select'
    CATEGORY = 'audio/YuE2'
    DESCRIPTION = 'Caps Standard semantic AR to the adapted score, with small musical tail; never fills max_duration automatically.'

    def select(self, abc, max_duration=190.0):
        duration, report = generation_budget(abc,max_duration=max_duration)
        print('[YuE2 Auto Vocal PRO][DURATION CONTRACT] '+json.dumps(report,ensure_ascii=False))
        return float(duration),json.dumps(report,ensure_ascii=False,indent=2)


NODE_CLASS_MAPPINGS = {'YuE2AdaptiveDurationBudget':YuE2AdaptiveDurationBudget}
NODE_DISPLAY_NAME_MAPPINGS = {'YuE2AdaptiveDurationBudget':'YuE2 · Adaptive Duration Budget'}
