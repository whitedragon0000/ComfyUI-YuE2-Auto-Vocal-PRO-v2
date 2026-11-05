"""Canonical ABC-to-Lyrics section contract for native YuE2 scores.

Every score section has exactly one lyrics section in musical order, including
silent/instrumental sections. Only user-provided words may occupy vocal blocks.
This is a deterministic symbolic contract, NOT a guarantee of acoustic fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc
from .lyric_contract import _tokens, automatic_lyrics
from .alignment import lyric_parts

HEADING = re.compile(r'^% (.+)$')


@dataclass(frozen=True)
class NativeSection:
    label: str
    first_bar: int
    after_bar: int
    vocal_onsets: int


def native_sections(abc: str) -> list[NativeSection]:
    """Read heading boundaries, native bar spans, and actual vocal note events."""
    score = parse_abc(abc)
    lines = abc.splitlines()
    staff = score.voices['Vocal']
    indices = sorted(score.music_lines)
    if len(indices) % 2 or any(score.music_lines[indices[n]] != 'Vocal' or
                                  score.music_lines[indices[n+1]] != 'Ins'
                                  for n in range(0, len(indices), 2)):
        raise ValueError('Native ABC does not contain properly paired voices')
    result = []
    cursor = 0
    preceding = 7
    for i in range(0, len(indices), 2):
        vi, ii = indices[i:i+2]
        labels = [m.group(1).strip() for index in range(preceding+1, vi)
                  if (m := HEADING.match(lines[index]))]
        count = sum(int(m.group(1) or '1') if
                    (m := re.fullmatch(r'Z([2-4])?', bar.strip())) else 1
                    for bar in lines[vi].rstrip('|').split('|'))
        if count == 0 or cursor + count > len(staff.bars):
            raise ValueError('Native score group and parsed measures diverge')
        start = staff.bars[cursor][0]
        stop_bar = staff.bars[cursor+count-1]
        stop = stop_bar[0] + stop_bar[1]
        notes = sum(start <= n[0] < stop for n in staff.notes)
        if labels or not result:
            # Multiple adjacent comments: the last is the label for this group.
            # A preceding empty comment is not an additional music section.
            result.append(NativeSection(labels[-1] if labels else 'Verse', cursor,
                                        cursor+count, notes))
        else:
            prev = result[-1]
            result[-1] = NativeSection(prev.label, prev.first_bar,
                                       cursor+count, prev.vocal_onsets+notes)
        cursor += count
        preceding = ii
    if cursor != len(staff.bars):
        raise ValueError('Native section groups do not cover the complete score')
    return result


def render_native_lyrics(abc: str, target_lyrics: str) -> tuple[str, dict]:
    """Return a one-to-one sectioned lyric document for a valid native ABC.

    The musical score is authoritative for section order, including intros,
    interludes, and outros. User words remain unchanged and in the same order.
    """
    sections = native_sections(abc)
    if not sections:
        raise ValueError('No native score sections to align lyrics')
    before, _ = _tokens(target_lyrics)
    if not before:
        raise ValueError('A sung score requires user target words; do not inject reference transcripts')
    # A previously materialized one-block-per-section document (including
    # deliberately empty vocal passages) is already a complete native lyric
    # contract. Do not feed it back into the allocator, which drops empty
    # blocks and would otherwise reject a perfectly valid cover form.
    _, existing = lyric_parts(target_lyrics)
    if existing and existing[-1].tag and existing[-1].tag.casefold() == 'end' and not existing[-1].sung_lines:
        existing = existing[:-1]
    from .alignment import _canonical
    if (len(existing) == len(sections) and
        all(p.tag and _canonical(p.tag) == _canonical(s.label)
            for p, s in zip(existing, sections))):
        return materialize_native_blocks(abc, target_lyrics)
    allocated, details = automatic_lyrics(abc, target_lyrics)
    if details['status'] != 'planned':
        raise ValueError('Lyrics could not be allocated to vocal sections: ' + details['reason'])
    return materialize_native_blocks(abc, allocated)


def materialize_native_blocks(abc: str, allocated: str) -> tuple[str, dict]:
    """Render every native section without re-allocating already planned words."""
    sections = native_sections(abc)
    before, _ = _tokens(allocated)
    _, parts = lyric_parts(allocated)
    populated = [p for p in parts if p.sung_lines]
    sung_sections = [s for s in sections if s.vocal_onsets]
    tagged = [p for p in parts if p.tag and p.tag.casefold() != 'end']
    from .alignment import _canonical
    # The native score can contain vocal passages with zero assigned words.
    # Empty blocks are legitimate; *silently inventing repeated words is not*.
    # Prefer exact section-by-section mapping where it is already present.
    if (len(tagged) == len(sections) and
        all(_canonical(p.tag) == _canonical(s.label)
            for p, s in zip(tagged, sections))):
        if any(p.sung_lines and not s.vocal_onsets for p, s in zip(tagged, sections)):
            raise ValueError('Words assigned to a score section with no vocal notes')
        vocal_text = [p.sung_lines for p, s in zip(tagged, sections) if s.vocal_onsets]
    elif (len(tagged) == len(sung_sections) and
          all(_canonical(p.tag) == _canonical(s.label)
              for p, s in zip(tagged, sung_sections))):
        vocal_text = [p.sung_lines for p in tagged]
    elif len(populated) == len(sung_sections):
        vocal_text = [p.sung_lines for p in populated]
    else:
        raise ValueError(f'Native lyrics section mismatch: {len(populated)} populated blocks, '
                         f'{len(tagged)} tagged blocks, {len(sung_sections)} vocal score sections')
    blocks = []
    vocal_idx = 0
    for s in sections:
        header = s.label.strip() or 'Verse'
        if any(x in header for x in '\n\r[]'):
            raise ValueError('Invalid ABC section heading for Lyrics: '+repr(header))
        if s.vocal_onsets:
            phrase_lines = vocal_text[vocal_idx]
            vocal_idx += 1
            blocks.append('['+header+']\n'+'\n'.join(phrase_lines))
        else:
            blocks.append('['+header+']')
    output = '\n\n'.join(blocks).rstrip()+'\n'
    after, _ = _tokens(output)
    if before != after:
        raise ValueError('Lyrics section contract changed user words')
    # Native YuE2 may still vocalize in sections whose score has vocal notes
    # but whose target text is empty. This is a real musical limitation, never
    # report it as certified lyric fidelity.
    blank_vocal = sum(not chunk for chunk in vocal_text)
    return output, {'section_contract': 'native_score_one_to_one',
                    'score_sections': len(sections),
                    'vocal_sections': len(sung_sections),
                    'empty_instrumental_sections': sum(not s.vocal_onsets for s in sections),
                    'unfilled_vocal_sections': blank_vocal,
                    'unfilled_vocal_risk': ('The ABC has vocal notes without user lyrics; YuE2 may invent or repeat words'
                                            if blank_vocal else None),
                    'word_sequence_identical': True,
                    'target_words': len(before)}
