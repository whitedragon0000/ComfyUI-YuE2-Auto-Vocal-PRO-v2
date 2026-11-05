"""Conservative instrumental continuation for source scores with very short lyrics.

This module does *not* synthesize new music or perform phoneme-to-note alignment.
When a source score contains substantially more vocal notes than the words can
plausibly occupy, it may revoice whole trailing bars to the instrumental staff.
No note, chord, timestamp, tempo or meter is discarded. Inability to prove
these invariants means the caller must use the native YuE2 ABC planner.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass

from .engine import is_only_rests, split_bars
from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc, AbcError
from .alignment import _syllable_estimate
from .lyric_contract import _tokens

_NOTE = re.compile(r'(?:\^\^|__|\^|_|=)?[A-Ga-g][,\']*(?:[0-9]+)?(?:-)?')


@dataclass(frozen=True)
class ContinuationResult:
    abc: str
    transferred_bars: int
    retained_vocal_onsets: int
    status: str


def _expanded(bars: list[str]) -> list[str]:
    result = []
    for bar in bars:
        m = re.fullmatch(r'Z([1-9][0-9]*)?', bar)
        if m:
            result.extend(['Z'] * int(m.group(1) or '1'))
        else:
            result.append(bar)
    return result


def _events(score):
    notes = collections.Counter()
    chords = collections.Counter()
    for voice in score.voices.values():
        for note in voice.notes:
            notes[tuple(note)] += 1
        for chord in voice.chords:
            chords[tuple(chord)] += 1
    return notes, chords


def instrumental_continuation(abc: str, lyrics: str, *, max_duration: float = 0) -> ContinuationResult:
    """Move unused *complete* vocal bars to the Ins staff when it is silent.

    Only safe native two-voice scores with a single text block are eligible.
    Chords and notes are preserved as sounding MIDI events (including times).
    This makes the song continue instrumentally after the shorter text, rather
    than leaving voiced ABC bars for YuE2 to fill with invented words.
    """
    original = parse_abc(abc)
    if max_duration > 0 and float(original.voices['Vocal'].time * 60 / original.bpm) > max_duration * 1.05:
        return ContinuationResult(abc, 0, 0, 'score_exceeds_budget')
    words, parts = _tokens(lyrics)
    blocks = [p for p in parts if p.sung_lines]
    if not words or len(blocks) != 1:
        return ContinuationResult(abc, 0, 0, 'multi_section_or_empty_lyrics')
    syllables = _syllable_estimate([' '.join(words)])
    # A little melisma is normal (two-ish notes per syllable). Select whole
    # *bars*, not one note per syllable; music boundaries remain grammatical.
    target_onsets = max(2, round(syllables * 1.8))
    lines = abc.splitlines()
    indices = sorted(original.music_lines)
    if len(indices) % 2 or any(original.music_lines[indices[i]] != 'Vocal' or
                               original.music_lines[indices[i+1]] != 'Ins'
                               for i in range(0, len(indices), 2)):
        return ContinuationResult(abc, 0, 0, 'unsupported_score_layout')
    changed = lines[:]
    remaining, transferred = target_onsets, 0
    retained = 0
    for i in range(0, len(indices), 2):
        vi, ii = indices[i], indices[i + 1]
        vb = _expanded(split_bars(lines[vi])); ib = _expanded(split_bars(lines[ii]))
        if len(vb) != len(ib):
            return ContinuationResult(abc, 0, 0, 'unaligned_staff_bars')
        vb_new, ib_new = [], []
        for vocal_bar, ins_bar in zip(vb, ib):
            count = len(_NOTE.findall(re.sub(r'"[^"\n]*"', '', vocal_bar)))
            if count == 0 or remaining > 0:
                vb_new.append(vocal_bar); ib_new.append(ins_bar)
                retained += count
                remaining -= count
                continue
            if not is_only_rests(ins_bar) or '[K:' in vocal_bar:
                return ContinuationResult(abc, 0, 0, 'instrumental_staff_occupied_or_key_change')
            vb_new.append('Z')
            ib_new.append(vocal_bar)
            transferred += 1
        changed[vi] = '|'.join(vb_new) + '|'
        changed[ii] = '|'.join(ib_new) + '|'
    if not transferred:
        return ContinuationResult(abc, 0, retained, 'nothing_to_transfer')
    candidate = '\n'.join(changed) + ('\n' if abc.endswith('\n') else '')
    try:
        check = parse_abc(candidate)
    except AbcError:
        return ContinuationResult(abc, 0, 0, 'candidate_invalid')
    if (original.bpm != check.bpm or original.unit != check.unit or
        any(original.voices[v].bars != check.voices[v].bars or
            original.voices[v].time != check.voices[v].time for v in ('Vocal', 'Ins')) or
        _events(original) != _events(check)):
        return ContinuationResult(abc, 0, 0, 'event_conservation_failed')
    return ContinuationResult(candidate, transferred, retained, 'instrumental_tail_arranged')
