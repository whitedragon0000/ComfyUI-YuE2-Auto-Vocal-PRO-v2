"""Revoice *unfilled* native YuE2 vocal groups as instrumental melody.

When a short replacement text occupies a single chorus in a long song,
leaving 30+ vocal groups with no Lyrics invites uncontrolled singing.  A
valid original lead sheet usually holds the sole melody on Vocal and rests
on Ins after Auto Vocal promotion.  For those groups only, move the exact
musical events to Ins and use its original whole-measure rests on Vocal.

Never delete notes, change tempo, split ties, or mix competing polyphony.
If the staff is not safely interchangeable, leave it and disclose the risk.
"""
from __future__ import annotations

from collections import Counter

from .section_contract import native_sections
from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc
from .engine import is_only_rests
from .vocal_melody_engine import _expanded_bar_count


def _event_inventory(score):
    notes = Counter()
    chords = Counter()
    for voice in score.voices.values():
        for note in voice.notes:
            notes[tuple(note)] += 1
        for chord in voice.chords:
            chords[tuple(chord)] += 1
    return notes, chords


def revoice_empty_sections(abc: str, empty_sections: list[int]):
    before = parse_abc(abc)
    sections = native_sections(abc)
    if not empty_sections:
        return abc, {'action':'none', 'groups_revoiced':0, 'sections_revoiced':0,
                     'sections_not_revoiced':[]}
    targets = set(empty_sections)
    lines=abc.splitlines()
    indices=sorted(before.music_lines)
    pending=[]
    bar_cursor=0
    skipped=[]
    shifted_sections=set()
    for i in range(0,len(indices),2):
        vi,ii=indices[i:i+2]
        count=_expanded_bar_count(lines[vi])
        section_index=next((idx for idx,s in enumerate(sections)
                            if s.first_bar<=bar_cursor<s.after_bar),None)
        bar_cursor += count
        if section_index not in targets:
            continue
        vocal_music,ins_music=lines[vi].strip(),lines[ii].strip()
        if ('-' in vocal_music or '-' in ins_music or
            not is_only_rests(ins_music) or
            is_only_rests(vocal_music) or
            '[K:' in vocal_music or '[M:' in vocal_music or
            '[K:' in ins_music or '[M:' in ins_music):
            skipped.append(section_index)
            continue
        pending.append((vi,ii,section_index))
    # Apply each candidate incrementally and validate the entire two-staff
    # score; any invalid transform is rolled back, not guessed.
    moved=0
    for vi,ii,section_index in pending:
        prior_vi,prior_ii=lines[vi],lines[ii]
        lines[vi],lines[ii]=prior_ii,prior_vi
        candidate='\n'.join(lines)+'\n'
        try:
            after=parse_abc(candidate)
            if (after.voices['Vocal'].bars != before.voices['Vocal'].bars or
                after.voices['Ins'].bars != before.voices['Ins'].bars or
                after.voices['Vocal'].time != before.voices['Vocal'].time or
                after.voices['Ins'].time != before.voices['Ins'].time or
                _event_inventory(after) != _event_inventory(before)):
                raise ValueError('Native notes/chords/measure-grid invariant failed')
        except (ValueError,IndexError,KeyError):
            lines[vi],lines[ii]=prior_vi,prior_ii
            skipped.append(section_index)
            continue
        moved+=1
        shifted_sections.add(section_index)
    result='\n'.join(lines)+'\n' if moved else abc
    still_unfilled=sorted(targets-shifted_sections)
    return result, {'action':'revoice_empty_vocal_sections',
                    'groups_revoiced': moved, 'sections_revoiced':len(shifted_sections),
                    'sections_not_revoiced':still_unfilled,
                    'note_events_preserved': True,
                    'warning': ('Unfilled passages moved to the instrumental staff; '
                                'YuE2 acoustic speech/lyrics compliance is not guaranteed.'
                                if moved else 'Unfilled vocal passages could not be safely revoiced.')}
