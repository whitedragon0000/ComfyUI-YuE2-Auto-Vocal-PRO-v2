"""Auditable, non-destructive lyrics-to-ABC section planner.

This is NOT a phoneme aligner. It never rewrites notes, rests, meter, key,
chords or timing; for unstructured lyrics it can distribute complete lines
across the actual vocal score sections using a globally optimal DP. There is
no silent lyric invention, deletion or repetition.
"""
from __future__ import annotations

import json
import math
from .alignment import _syllable_estimate, align_lyrics, lyric_parts, score_runs

POLICIES = ('adaptive', 'melody_priority', 'lyrics_priority')


def _solve_partitions(weights: list[int], capacities: list[int]) -> list[int]:
    """Exact DP, assigning >=1 whole text lines to each vocal section.

    Objective: minimize total squared difference between cumulative text
    syllable share and cumulative score-onset share. Monotone line order is
    guaranteed. Deterministic earliest-boundary tie-break.
    """
    n, k = len(weights), len(capacities)
    if not (1 <= k <= n):
        raise ValueError('Cannot allocate at least one lyric line to each vocal section')
    total = max(1, sum(weights))
    cap_total = max(1, sum(capacities))
    pref = [0]
    for w in weights:
        pref.append(pref[-1] + w)
    targets = [0.0]
    acc = 0
    for cap in capacities:
        acc += cap
        targets.append(acc / cap_total)
    # state (cost, ordered boundary tuple). No approximated greedy chunking.
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for section in range(1, k):
        nxt = {}
        for prev, (cost, boundaries) in dp.items():
            for end in range(prev + 1, n - (k-section) + 1):
                error = ((pref[end] / total) - targets[section])**2
                candidate = (cost + error, boundaries + (end,))
                if end not in nxt or candidate < nxt[end]:
                    nxt[end] = candidate
        dp = nxt
    best = min((cost, bounds) for _, (cost, bounds) in dp.items()) if k > 1 else (0.0, ())
    return [0, *best[1], n]


def plan(abc: str, lyrics: str, policy: str = 'adaptive') -> tuple[str, str, str]:
    if policy not in POLICIES:
        raise ValueError(f'Unsupported lyric-fit policy: {policy}')
    runs = score_runs(abc)
    _, parsed = lyric_parts(lyrics)
    parts = [p for p in parsed if p.sung_lines]
    lyric_lines = [line for part in parts for line in part.sung_lines]
    input_syllables = _syllable_estimate(lyric_lines)
    has_user_tags = any(p.tag is not None for p in parts)
    meta = {
        'policy': policy,
        'vocal_sections': len(runs),
        'lyrics_sections': len(parts),
        'lyric_lines': len(lyric_lines),
        'approx_syllables': input_syllables,
        'vocal_note_onsets': sum(r.onsets for r in runs),
        'score_changed': False,
        'invented_words': False,
        'repeated_words': False,
        'action': 'preserved',
        'warnings': [],
    }
    if not runs or not lyric_lines:
        meta['warnings'].append('Missing vocal score or sung text; cannot establish section fit.')
        return abc, lyrics, json.dumps(meta, ensure_ascii=False, sort_keys=True)
    if len(parts) == len(runs):
        # Existing one-to-one section labels are authoritative.
        meta['action'] = 'existing_sections'
        return abc, lyrics, json.dumps(meta, ensure_ascii=False, sort_keys=True)
    if policy == 'melody_priority':
        meta['warnings'].append('Lyric section count differs; melody and user lyrics preserved unchanged.')
        return abc, lyrics, json.dumps(meta, ensure_ascii=False, sort_keys=True)
    if has_user_tags and policy == 'adaptive':
        meta['warnings'].append('Explicit user section tags preserved; select lyrics_priority to allow automatic repartition.')
        return abc, lyrics, json.dumps(meta, ensure_ascii=False, sort_keys=True)
    if len(lyric_lines) > 240 or len(runs) > 80:
        meta['warnings'].append('Arrangement exceeds bounded alignment planner size; preserving original lyrics.')
        return abc, lyrics, json.dumps(meta, ensure_ascii=False, sort_keys=True)
    if len(lyric_lines) < len(runs):
        meta['warnings'].append(f'Only {len(lyric_lines)} lyric lines for {len(runs)} vocal sections; will not invent/repeat lines.')
        return abc, lyrics, json.dumps(meta, ensure_ascii=False, sort_keys=True)
    # A single user section with multiple lines can be split across actual
    # vocal sections. Explicitly tagged lyrics are resectioned only on opt-in.
    syllables = [_syllable_estimate([line]) for line in lyric_lines]
    boundaries = _solve_partitions(syllables, [max(1,r.onsets) for r in runs])
    results = []
    allocation = []
    for i, run in enumerate(runs):
        start,end = boundaries[i:i+2]
        segment = lyric_lines[start:end]
        label = run.label if run.label and not run.label.startswith('<') else 'Verse'
        results.append('[' + label.title() + ']\n' + '\n'.join(segment))
        allocation.append({'section': i+1, 'label': label, 'lines': len(segment),
                           'syllables': sum(syllables[start:end]), 'note_onsets': run.onsets})
    output = '\n\n'.join(results) + ('\n' if lyrics.endswith('\n') else '')
    # Independent input-to-output invariants: lexical content & sequence.
    _, output_parts = lyric_parts(output)
    flattened = [line for part in output_parts for line in part.sung_lines]
    if flattened != lyric_lines:
        raise AssertionError('Lyric resection lost or duplicated a sung line')
    meta['action'] = 'repartitioned_whole_lines'
    meta['allocation'] = allocation
    meta['warnings'].append('Line-level alignment only; this does not enforce phoneme-to-note timing.')
    return abc, output, json.dumps(meta, ensure_ascii=False, sort_keys=True)
