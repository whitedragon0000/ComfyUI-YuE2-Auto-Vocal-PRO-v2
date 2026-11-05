"""Automatic, deterministic lyric/score contract for YuE2.

Uses the *native* score produced by SheetSage2 or YuE2GenerateABC; it does
not claim to implement the missing trained phoneme-to-note alignment model.
Every input word is preserved exactly and assigned to a vocal score section.
Incompatible melodic plans are marked for native ABC regeneration, never
silently retimed, repeated, or truncated.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from .alignment import _syllable_estimate, lyric_parts, score_runs


@dataclass(frozen=True)
class LyricFitness:
    score_sections: int
    text_sections: int
    words: int
    syllables_estimated: int
    note_onsets: int
    notes_per_syllable: float
    compatible: bool
    reason: str
    per_section: tuple = ()
    duration_seconds: float | None = None
    duration_ratio: float | None = None


def _tokens(lyrics: str):
    """Use word boundaries only, preserving spelling, apostrophes and punctuation."""
    _, parts = lyric_parts(lyrics)
    return [word for part in parts for line in part.sung_lines
            for word in line.split() if word.strip()], parts


def _word_syllables(word: str) -> int:
    return max(1, _syllable_estimate([word]))


def _estimated_duration(abc: str):
    """Use the upstream score parser, not a pipes≈bars duration estimate."""
    try:
        from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc
        score = parse_abc(abc)
        return float(score.voices['Vocal'].time * 60 / score.bpm)
    except (ValueError, KeyError, ZeroDivisionError, TypeError):
        return None


def _allocate_by_syllables(items: list[str], capacities: list[int]) -> list[list[str]]:
    """Allocate whole words or lyric lines by vowel-group mass in O(N+K).

    The input tokens and their order are invariant. No hidden repetition or
    word boundary edits. When K>N the remaining sections are explicit gaps.
    """
    n, k = len(items), len(capacities)
    if not k:
        return []
    if n < k:
        return [[items[i]] if i < n else [] for i in range(k)]
    masses = [max(1, _syllable_estimate([item])) for item in items]
    prefix = [0]
    for m in masses:
        prefix.append(prefix[-1] + m)
    total_capacity = sum(max(1, c) for c in capacities)
    seen = 0
    cursor = 0
    result = []
    for i, c in enumerate(capacities[:-1]):
        seen += max(1, c)
        goal = prefix[-1] * seen / total_capacity
        lo = cursor + 1
        hi = n - (k - i - 1)
        nxt = lo
        while nxt < hi and prefix[nxt] < goal:
            nxt += 1
        if nxt > lo and abs(prefix[nxt-1] - goal) < abs(prefix[nxt] - goal):
            nxt -= 1
        result.append(items[cursor:nxt])
        cursor = nxt
    result.append(items[cursor:])
    return result


def measure(abc: str, lyrics: str, *, max_duration: float | None = None) -> LyricFitness:
    """Score/lyrics fitness with BOTH global and per-vocal-section gates.

    Scores are only a *planning* signal; model acoustics remain probabilistic.
    The duration is obtained from the real ABC event parser if it supports
    the score; unsupported ABC is diagnosed separately by the caller.
    """
    runs = score_runs(abc)
    words, parts = _tokens(lyrics)
    populated = [p for p in parts if p.sung_lines]
    syllables = _syllable_estimate([' '.join(words)]) if words else 0
    onsets = sum(max(0, r.onsets) for r in runs)
    ratio = onsets / max(1, syllables)
    # Explicit user-supplied sections are intentional: check their real local
    # syllable density, not an optimistic reassignment that would hide a mismatch.
    if len(populated) == len(runs) and all(p.tag for p in populated):
        planned = [[w for line in part.sung_lines for w in line.split()]
                   for part in populated]
    else:
        planned = _allocate_by_syllables(words, [max(1, r.onsets) for r in runs]) if runs else []
    per_section = []
    for run, assigned in zip(runs, planned):
        sy = sum(_word_syllables(w) for w in assigned)
        local_ratio = run.onsets / max(1, sy)
        per_section.append({'label': run.label, 'bars': run.bars,
                            'note_onsets': run.onsets, 'words': len(assigned),
                            'estimated_syllables': sy,
                            'notes_per_syllable': round(local_ratio, 3)})
    seconds = _estimated_duration(abc)
    budget_ratio = (seconds / max_duration if seconds is not None and max_duration and max_duration > 0 else None)
    if not words:
        reason = 'no_lyrics'
    elif not runs:
        reason = 'no_vocal_sections'
    elif len(words) < len(runs):
        reason = 'fewer_words_than_vocal_sections'
    elif budget_ratio is not None and budget_ratio > 1.05:
        reason = 'score_exceeds_music_budget'
    elif ratio > (4.0 if syllables <= 2 else 3.0):
        reason = 'sparse_lyrics_for_melody'
    elif ratio < .55:
        reason = 'too_many_syllables_for_melody'
    elif any(x['notes_per_syllable'] > 4.0 or x['notes_per_syllable'] < .4
             for x in per_section):
        reason = 'per_section_syllable_mismatch'
    else:
        reason = 'compatible'
    return LyricFitness(len(runs), len(populated), len(words), syllables,
                        onsets, round(ratio, 3), reason == 'compatible', reason,
                        tuple(per_section), seconds, round(budget_ratio, 4) if budget_ratio is not None else None)


def _split_weighted(items: list[str], capacities: list[int]) -> list[list[str]]:
    """O(n+k) order-preserving allocation; no hard split inside words.

    Computes cumulative integer quotas, giving each vocal section at least
    one word whenever N >= K. This is not a phonetic timing model.
    """
    k, n = len(capacities), len(items)
    if not k or not n:
        return [[] for _ in capacities]
    # If fewer words than sections, leave later sections empty but signal it
    # at the caller; the automatic scorer will request re-composition.
    if n < k:
        return [[word] if i < n else [] for i, word in enumerate(items + [''] * (k-n))]
    capacity_total = max(1, sum(max(1, c) for c in capacities))
    cumulative = 0
    prev = 0
    boundaries = [0]
    for idx, cap in enumerate(capacities[:-1]):
        cumulative += max(1, cap)
        proposed = round(n * cumulative / capacity_total)
        end = max(prev + 1, min(n - (k-idx-1), proposed))
        boundaries.append(end)
        prev = end
    boundaries.append(n)
    return [items[boundaries[i]:boundaries[i+1]] for i in range(k)]


def automatic_lyrics(abc: str, lyrics: str) -> tuple[str, dict]:
    """Produce one lyric block per actual vocal score run, conserving all words.

    If explicit sections match the score count, preserve their line content.
    Otherwise use whole text lines where possible, and whole words if there
    are fewer lines than score sections. Instrumental-only tags are excluded.
    """
    runs = score_runs(abc)
    words, parts = _tokens(lyrics)
    populated = [p for p in parts if p.sung_lines]
    fitness = measure(abc, lyrics)
    report = {
        'scheme': 'automatic_score_lyrics_v1',
        'lyrics_words': fitness.words,
        'score_vocal_sections': len(runs),
        'input_lyric_sections': len(populated),
        'notes_per_estimated_syllable': fitness.notes_per_syllable,
        'native_abc_recommended': not fitness.compatible,
        'reason': fitness.reason,
        'per_section': list(fitness.per_section),
        'estimated_score_seconds': fitness.duration_seconds,
        'guarantee': 'input word sequence preserved; no acoustic articulation guarantee',
    }
    if not words or not runs:
        report['status'] = 'unplannable'
        return lyrics, report
    # A section-accurate native plan can be passed back by the Reference
    # composer. It must be idempotent: never redistribute its words or reject
    # empty sung sections just because the text was already materialized.
    from .section_contract import native_sections, materialize_native_blocks
    from .alignment import _canonical
    native_grid = native_sections(abc)
    original_blocks = [p for p in parts if p.tag and p.tag.casefold() != 'end']
    if (len(original_blocks) == len(native_grid) and
        all(_canonical(p.tag) == _canonical(s.label)
            for p, s in zip(original_blocks,native_grid))):
        materialized, coverage = materialize_native_blocks(abc,lyrics)
        report['status'] = 'planned'
        report['allocation'] = 'native_blocks_idempotent'
        report['native_section_contract'] = coverage
        report['section_word_counts'] = [sum(len(line.split()) for line in p.sung_lines)
                                          for p,s in zip(original_blocks,native_grid) if s.vocal_onsets]
        report['empty_vocal_sections'] = [i+1 for i,(p,s) in enumerate(zip(original_blocks,native_grid))
                                           if s.vocal_onsets and not p.sung_lines]
        return materialized, report
    # Multiple explicitly named sections encode a deliberate musical order.
    # The old allocation silently spread their lines by note density when the
    # section count differed, e.g. [Verse]/[Chorus] -> two unrelated Verse
    # passages. Unlike untagged prose, this is not safe to guess.
    from .alignment import _canonical
    named = [p for p in populated if p.tag]
    # Every explicitly tagged passage names a musical section. In particular,
    # a single [Chorus] MUST NEVER be spread over a prefix containing only
    # Verse sections: doing so was silently changing the song requested by
    # the user and could result in unwanted source-like singing.
    same_named_section = (len(named) == len(populated) == 1 and
                          all(_canonical(run.label) == _canonical(named[0].tag) for run in runs))
    if len(populated) != len(runs) and len(named) == len(populated) and named and not same_named_section:
        raise ValueError('Explicit Lyrics sections require a matching score excerpt: '
                         f'{len(populated)} tagged passages versus {len(runs)} vocal '
                         'sections. The planner must first select the named '
                         'music sections, not relabel or redistribute the words.')
    if len(populated) != len(runs) and len(named) >= 2 and len({_canonical(p.tag) for p in named}) > 1:
        raise ValueError('Explicit Lyrics sections cannot be aligned to the source score: '
                         f'{len(populated)} tagged lyric passages vs {len(runs)} vocal score '
                         'sections. Supply one Lyrics block per sung section in source order '
                         'or request an explicitly shorter score excerpt.')
    if len(populated) == len(runs):
        # User-authored labels are semantic anchors, not decorations. The older allocator used
        # to relabel the musical score implicitly when the counts matched;
        # e.g. [Chorus] was sung over the first % verse group. That destroys
        # the source song's phrasing even though all word hashes still match.
        if all(part.tag for part in populated):
            from .alignment import _canonical
            for index, (run, part) in enumerate(zip(runs, populated), 1):
                if _canonical(run.label) != _canonical(part.tag):
                    raise ValueError(
                        f'Lyrics section {index} [{part.tag}] does not match source '
                        f'ABC section %{run.label}; keep the source score order, '
                        'or generate an explicitly section-selected excerpt.')
        allocated = [p.sung_lines[:] for p in populated]
        report['allocation'] = 'explicit_sections'
        # Preserve the user's exact section labels, punctuation and spacing;
        # score headings are synchronized separately by align_lyrics().
        if all(p.tag is not None for p in populated):
            report['status'] = 'planned'
            report['empty_vocal_sections'] = []
            report['section_word_counts'] = [sum(len(line.split()) for line in g) for g in allocated]
            from .section_contract import materialize_native_blocks
            materialized, coverage = materialize_native_blocks(abc, lyrics)
            report['native_section_contract'] = coverage
            return materialized, report
    else:
        lines = [line for part in populated for line in part.sung_lines]
        caps = [max(1, r.onsets) for r in runs]
        if len(lines) >= len(runs):
            allocated = _allocate_by_syllables(lines, caps)
            report['allocation'] = 'whole_lines'
        else:
            allocated = [[' '.join(group)] if group else [] for group in _allocate_by_syllables(words, caps)]
            report['allocation'] = 'whole_words'
    blocks = []
    for run, sung_lines in zip(runs, allocated):
        # Keep score labels authoritative: compositing will reuse these labels.
        name = (run.label or '').strip()
        if not name or name.startswith('<'):
            name = 'Verse'
        blocks.append('[' + name.title() + ']\n' + '\n'.join(sung_lines))
    result = '\n\n'.join(blocks).rstrip() + '\n'
    result_words, _ = _tokens(result)
    if result_words != words:
        raise AssertionError('Automatic lyric assignment added/dropped/reordered words')
    report['status'] = 'planned'
    report['empty_vocal_sections'] = [i + 1 for i, group in enumerate(allocated) if not group]
    report['section_word_counts'] = [sum(len(line.split()) for line in group) for group in allocated]
    from .section_contract import materialize_native_blocks
    materialized, coverage = materialize_native_blocks(abc, result)
    report['native_section_contract'] = coverage
    return materialized, report


def musical_policy_report(abc: str, lyrics: str) -> str:
    return json.dumps(measure(abc, lyrics).__dict__, ensure_ascii=False, sort_keys=True)


def with_terminal_end(lyrics: str) -> str:
    """Preserve sung words verbatim and add the official YuE2 end-of-song cue.

    nvmax/ComfyUI-YuE2's clean_yue2_lyrics() likewise makes [End] terminal.
    We deliberately avoid changing spelling, verse content or user punctuation.
    """
    if not lyrics or not lyrics.strip():
        return lyrics
    lines = lyrics.rstrip().splitlines()
    positions = [idx for idx, line in enumerate(lines)
                 if re.fullmatch(r'\s*\[End\]\s*', line, flags=re.I)]
    if positions and positions[-1] != len(lines) - 1:
        raise ValueError('A [End] marker appears before the final lyric line. '
                         'Move it to the very end to avoid premature song termination.')
    if positions:
        return lyrics
    return lyrics.rstrip() + '\n\n[End]\n'


def with_instrumental_tail(abc: str, lyrics: str) -> str:
    """Represent instrumental sections after the final sung section explicitly.

    ABC can contain many instrumental bars after a short lyric. The old lyric
    composer appended [End] immediately after the last sung word, making a
    premature YuE2 MUSIC_END plausible despite the continued instrumental score.
    The returned text preserves all sung words and adds only structural tags.
    """
    from .engine import SECTION_COMMENT
    from .alignment import score_runs
    runs = [r for r in score_runs(abc) if r.onsets > 0]
    if not runs:
        return lyrics
    last_vocal_position = max((r.heading_index if r.heading_index is not None else -1) for r in runs)
    sections = []
    for index, line in enumerate(abc.splitlines()):
        if index <= last_vocal_position:
            continue
        match = SECTION_COMMENT.match(line)
        if match:
            sections.append(match.group(1).strip())
    # Only explicitly instrumental named sections. Do not hallucinate a new
    # instrumental passage simply because one trailing bar contains rests.
    recognized = {'interlude','intermezzo','instrumental','instrumental break','outro',
                  'ending','solo','music break','breakdown','instrumental outro'}
    suffix = [label for label in sections if label.casefold() in recognized]
    if not suffix:
        return lyrics
    cleaned = lyrics.rstrip()
    if re.search(r'(?im)^\s*\[End\]\s*$', cleaned):
        lines = cleaned.splitlines()
        if lines and re.fullmatch(r'\s*\[End\]\s*', lines[-1], flags=re.I):
            cleaned = '\n'.join(lines[:-1]).rstrip()
    # An already-rendered tail must remain idempotent during ref composition.
    present = []
    for line in reversed(cleaned.splitlines()):
        if not line.strip(): continue
        tag = re.fullmatch(r'\[([^\]]+)\]', line.strip())
        if not tag: break
        present.insert(0, tag.group(1).strip())
    if len(present) >= len(suffix) and [x.casefold() for x in present[-len(suffix):]] == [x.casefold() for x in suffix]:
        return cleaned + '\n'
    return cleaned + '\n\n' + '\n\n'.join('['+label.title()+']' for label in suffix) + '\n'
