"""Non-invasive, symbolic pronunciation-pressure diagnostic.

This measures how much user text is assigned to each vocal section relative to
its measured note onsets and score time. It does NOT evaluate pronunciation,
phonemes, acoustic clarity, or the sung words. It does not modify ABC or Lyrics.
"""
from __future__ import annotations

from .section_contract import native_sections
from .alignment import lyric_parts, _syllable_estimate
from .lyric_contract import _tokens
from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc


def diction_pressure(abc: str, allocated_lyrics: str) -> dict:
    score = parse_abc(abc)
    sections = native_sections(abc)
    _, parts = lyric_parts(allocated_lyrics)
    tagged = [part for part in parts if part.tag and part.tag.casefold() != 'end']
    if len(tagged) != len(sections):
        return {'status': 'not_evaluable', 'reason': 'Lyrics and ABC are not one-to-one sections'}

    staff = score.voices['Vocal']
    entries = []
    risks = []
    for idx, (section, part) in enumerate(zip(sections, tagged)):
        lines = part.sung_lines
        words = len(_tokens('\n'.join(lines))[0]) if lines else 0
        syllables = _syllable_estimate(lines) if lines else 0
        start = staff.bars[section.first_bar][0]
        last = staff.bars[section.after_bar - 1]
        duration = max(0.0, float((last[0] + last[1] - start) * 60 / score.bpm))
        rate = words / duration if duration > 0 else 0.0
        syl_rate = syllables / duration if duration > 0 else 0.0
        onset_load = syllables / section.vocal_onsets if section.vocal_onsets else 0.0
        # An *advisory*: fast rap, spoken passages and melismas have different
        # legitimate envelopes; never treat these thresholds as hard gates.
        crowded = bool(words and (rate > 3.5 or syl_rate > 7.0 or onset_load > 7.0))
        entries.append({'index': idx, 'label': section.label,
                        'duration_seconds': round(duration, 3),
                        'vocal_note_onsets': section.vocal_onsets,
                        'assigned_words': words, 'estimated_syllables': syllables,
                        'words_per_second': round(rate, 3),
                        'estimated_syllables_per_second': round(syl_rate, 3),
                        'estimated_syllables_per_note_onset': round(onset_load, 3),
                        'possible_articulation_pressure': crowded})
        if crowded:
            risks.append(idx)
    return {'status': 'estimated_symbolic_only', 'section_count': len(entries),
            'risk_section_indices': risks, 'risk_section_count': len(risks),
            'sections': entries,
            'limitations': 'Not phoneme alignment or acoustic speech intelligibility; the native model can sing differently.'}
