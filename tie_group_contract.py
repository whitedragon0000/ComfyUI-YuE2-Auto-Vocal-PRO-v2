"""Preserve cross-group ABC ties when promoting instrumental melody to Vocal.

YuE2's native parser treats a trailing '-' as pending until the *next*
bar, which may belong to another `% section`/Vocal+Ins group. A group-by-group
promotion can put the pending note on Vocal and the continuation on Ins,
producing `tie enters a full-measure rest` in Vocal. Preflight complete
connected components of instrumental note ties before changing either staff.
No ties are removed and no notes are rewritten.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceGroup:
    start: int
    section: str
    vocal: str
    instrumental: str
    convertible: bool


def blocked_tie_groups(lines, section_policy, instrumental_tags, vocal_tags,
                       existing_vocal):
    from .engine import (VOICE_BODY, SECTION_COMMENT, classify, extract_music,
                         split_bars, is_note_content, CHORD)
    from .instrumental_mode import _count_measures
    groups = []
    section = ''
    i = 0
    while i < len(lines):
        m = SECTION_COMMENT.match(lines[i])
        if m:
            section = m.group(1).strip()
        if not (v := VOICE_BODY.match(lines[i])) or v.group(1).lower() != 'vocal':
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not VOICE_BODY.match(lines[j]) and not SECTION_COMMENT.match(lines[j]):
            j += 1
        if j == len(lines) or not (ins := VOICE_BODY.match(lines[j])) or ins.group(1).lower() != 'ins':
            i = j
            continue
        k = j + 1
        while k < len(lines) and not VOICE_BODY.match(lines[k]) and not SECTION_COMMENT.match(lines[k]):
            k += 1
        _, vscore = extract_music(lines[i+1:j])
        _, iscore = extract_music(lines[j+1:k])
        vbars, ibars = split_bars(vscore), split_bars(iscore)
        complex_chorded = (len(vbars) != len(ibars) and any(CHORD.search(x) for x in vbars))
        convertible = bool(vscore and iscore and
            classify(section, section_policy, vocal_tags, instrumental_tags) and
            (existing_vocal != 'preserve' or not any(is_note_content(x) for x in vbars)) and
            any(is_note_content(x) for x in ibars) and
            _count_measures(vscore) == _count_measures(iscore) and not complex_chorded)
        groups.append(SourceGroup(i, section, vscore, iscore, convertible))
        i = k
    # A native two-staff score carries two *independent* tie continuations.
    # Overwrite may remove Vocal events as well as promote Ins into Vocal;
    # protecting only Ins ties (the old implementation) is insufficient.
    # Build full connected components along BOTH original staff tie boundaries.
    # Each component must be converted atomically or not converted at all.
    def ends_in_tie(music):
        # Drop native bar separators and chord annotations, preserving the
        # final note's trailing '-' (a native ABC tie, not punctuation).
        clean = CHORD.sub('', music).strip().rstrip('|').rstrip()
        return clean.endswith('-')

    blocked = set()
    index = 0
    while index < len(groups):
        end = index
        while end + 1 < len(groups) and (
            ends_in_tie(groups[end].instrumental) or
            ends_in_tie(groups[end].vocal)
        ):
            end += 1
        chain = groups[index:end + 1]
        if len(chain) > 1 and not all(group.convertible for group in chain):
            blocked.update(group.start for group in chain)
        index = end + 1
    return blocked
