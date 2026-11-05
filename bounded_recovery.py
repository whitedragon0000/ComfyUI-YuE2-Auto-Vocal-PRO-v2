"""Native ABC excerpt selector for explicit duration limits.

This module does NOT perform phonetic alignment, impose syllable ratios, or
regenerate music. It selects parseable, contiguous source groups whose musical
section order is compatible with requested lyric headings. The sole hard time
constraint is the caller's exact `max_duration`.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop
import math

from .alignment import _canonical, lyric_parts, _expanded_bar_count
from .lyric_contract import _tokens
from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc


@dataclass(frozen=True)
class Group:
    label: str
    body: tuple[str, ...]
    heading: bool
    seconds: float
    vocal_onsets: int


def _groups(abc: str):
    """Get parsed Vocal/Ins pairs, retaining local M:/K:/L: directives.

    The native parser identifies the music lines. Voice markers are found by
    walking BACK over local metadata, rather than assuming `V: Vocal` is the
    immediately preceding line. This is important after a metre/key change.
    """
    score = parse_abc(abc)
    lines = abc.splitlines()
    indices = sorted(score.music_lines)
    if len(indices) % 2 or not indices:
        return (), score
    if any(score.music_lines[indices[i]] != 'Vocal' or score.music_lines[indices[i + 1]] != 'Ins'
           for i in range(0, len(indices), 2)):
        return (), score
    from .engine import VOICE_BODY, BODY_META
    groups = []
    prior_end = 7
    bar_cursor = 0
    previous_label = 'Verse'
    # Sorted once: counting note events by scanning the entire vocal track for
    # every group was O(groups × notes), dominating large-score processing.
    vocal_onset_times = sorted(note[0] for note in score.voices['Vocal'].notes)
    for i in range(0, len(indices), 2):
        vi, ii = indices[i], indices[i + 1]
        vocal_start = vi - 1
        while vocal_start > prior_end and BODY_META.match(lines[vocal_start]):
            vocal_start -= 1
        ins_start = ii - 1
        while ins_start > vi and BODY_META.match(lines[ins_start]):
            ins_start -= 1
        if (vocal_start <= prior_end or ins_start <= vi or
            not (mv := VOICE_BODY.match(lines[vocal_start])) or mv.group(1).lower() != 'vocal' or
            not (mi := VOICE_BODY.match(lines[ins_start])) or mi.group(1).lower() != 'ins'):
            return (), score
        between_staffs = lines[vi + 1:ins_start]
        if any(x.strip() for x in between_staffs):
            return (), score
        before = lines[prior_end + 1:vocal_start]
        heading_lines = [x[1:].strip() for x in before if x.lstrip().startswith('%')]
        # Preserve directives immediately preceding the staff, if any.
        prefix_meta = [x for x in before if BODY_META.match(x)]
        if any(x.strip() and not x.lstrip().startswith('%') and not BODY_META.match(x)
               for x in before):
            return (), score
        label = heading_lines[-1] if heading_lines else previous_label
        count = _expanded_bar_count(lines[vi])
        if count < 1 or bar_cursor + count > len(score.voices['Vocal'].bars):
            return (), score
        span = score.voices['Vocal'].bars[bar_cursor:bar_cursor + count]
        start, stop = span[0][0], span[-1][0] + span[-1][1]
        onsets = (bisect_left(vocal_onset_times, stop)
                  - bisect_left(vocal_onset_times, start))
        body = prefix_meta + lines[vocal_start:ii + 1]
        groups.append(Group(label, tuple(body), bool(heading_lines),
                            float((stop - start) * 60 / score.bpm), onsets))
        previous_label = label
        bar_cursor += count
        prior_end = ii
    if bar_cursor != len(score.voices['Vocal'].bars):
        return (), score
    return tuple(groups), score


def _render(header, groups):
    lines = list(header)
    last = None
    for group in groups:
        if group.label != last:
            lines.append('% ' + group.label)
            last = group.label
        lines.extend(group.body)
    return '\n'.join(lines) + '\n'


def recover_score_for_lyrics(abc: str, lyrics: str, max_duration: float = 0.0,
                             *, minimum_ratio: float = 0.10, maximum_ratio: float = 12.0):
    """Select an existing source excerpt or return None.

    `minimum_ratio` and `maximum_ratio` remain in the call signature only for
    compatibility with older callers; they have no effect on musical validity.
    """
    groups, parsed = _groups(abc)
    words, parts = _tokens(lyrics)
    if not groups or not words:
        return None
    wanted = [_canonical(p.tag) for p in parts if p.sung_lines and p.tag]
    original_s = float(parsed.voices['Vocal'].time * 60 / parsed.bpm)
    cap = float(max_duration) if max_duration > 0 else float('inf')
    # Rank candidates lazily rather than building every O(N²) interval.
    # Positive group durations mean that, for any starting group, valid ends
    # under a hard duration cap form an interval. For explicit section tags,
    # the interval is limited to the last required section run.
    group_count = len(groups)
    normalized = [_canonical(g.label) for g in groups]
    run_end = [group_count] * group_count
    for i in range(group_count - 2, -1, -1):
        run_end[i] = run_end[i + 1] if normalized[i] == normalized[i + 1] else i + 1
    next_vocal = [group_count] * (group_count + 1)
    for i in range(group_count - 1, -1, -1):
        next_vocal[i] = i if groups[i].vocal_onsets else next_vocal[i + 1]
    elapsed = [0.0]
    for g in groups:
        elapsed.append(elapsed[-1] + g.seconds)

    # (-actual seconds, start, stop, min_stop). The tuple order retains the
    # previous ranking: longest music first; then earliest source position.
    def locally_summed_seconds(start, stop):
        # Match the original left-to-right addition order. Prefix subtraction
        # changes rounding and can silently select a different equal-length
        # section when the tempo yields repeating decimal bar lengths.
        total = 0.0
        for group in groups[start:stop]:
            total += group.seconds
        return total

    queue = []
    for start in range(group_count):
        if math.isfinite(cap):
            cap_end = bisect_right(elapsed, elapsed[start] + cap + 1e-6, lo=start + 1) - 1
        else:
            cap_end = group_count
        if cap_end <= start:
            continue
        if wanted:
            cursor = start
            valid = True
            for required_label in wanted:
                if cursor >= group_count or normalized[cursor] != required_label:
                    valid = False
                    break
                section_end = run_end[cursor]
                voiced = next_vocal[cursor]
                if voiced >= section_end:
                    valid = False
                    break
                cursor = section_end
            if not valid:
                continue
            min_stop = voiced + 1
            max_stop = min(section_end, cap_end)
        else:
            vocal_start = next_vocal[start]
            if vocal_start == group_count:
                continue
            min_stop = vocal_start + 1
            max_stop = cap_end
        if min_stop <= max_stop:
            heappush(queue, (-locally_summed_seconds(start, max_stop), start, max_stop, min_stop))

    found = None
    while queue:
        _negative_seconds, start, stop, min_stop = heappop(queue)
        # If a candidate cannot pass native parsing/lyrics mapping, consider
        # the next shorter candidate from this exact starting point.
        if stop > min_stop:
            next_stop = stop - 1
            heappush(queue, (-locally_summed_seconds(start, next_stop),
                             start, next_stop, min_stop))
        labels = []
        for g in groups[start:stop]:
            label = _canonical(g.label)
            if not labels or labels[-1] != label:
                labels.append(label)
        proposal = _render(abc.splitlines()[:8], groups[start:stop])
        try:
            test = parse_abc(proposal)
            if not test.voices['Vocal'].notes:
                continue
            duration = float(test.voices['Vocal'].time * 60 / test.bpm)
            if duration > cap + 1e-6:
                continue
            # Exactly the same section contract as the standard path.
            from .cover_contract import _allocate_text
            allocated, report = _allocate_text(proposal, lyrics)
            if _tokens(allocated)[0] != words or not report['words_identical_and_in_order']:
                continue
        except (ValueError, IndexError, KeyError, ZeroDivisionError):
            continue
        found = (duration, start, stop, proposal, tuple(labels))
        break
    if found is None:
        return None
    duration, start, stop, selection, labels = found
    return selection, {
        'action': 'source_score_contiguous_musical_excerpt',
        'source_duration_s': round(original_s, 3),
        'output_duration_s': round(duration, 3),
        'selected_group_range': [start, stop],
        'source_form_preserved': start == 0 and stop == len(groups),
        'target_words_preserved': True,
        'target_section_labels': wanted,
        'matched_score_sections': labels,
        'note_events_modified': False,
        'acoustic_lyrics_fidelity_guaranteed': False,
        'warning': 'Music is a contiguous excerpt, not the entire uploaded song.'
        if start or stop < len(groups) else None,
    }
