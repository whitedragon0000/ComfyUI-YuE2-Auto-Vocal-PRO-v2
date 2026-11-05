"""Instrumental-only score routing for the installed native two-voice YuE2 dialect.

Never sends the source/reference transcript into the new song. No sung lyrics
means no voice-promotion, even if the input contains a Vocal melody.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc

DIRECTIVES = frozenset({
    'instrumental', 'instrumental only', 'only instrumental', 'instrumental music',
    'strumentale', 'solo strumentale', 'no vocals', 'without vocals',
    'no lyrics', 'senza voce', 'senza vocali', 'senza testo',
})
END = '[Instrumental]\n\n[End]\n'


def instrumental_requested(lyrics: str | None) -> bool:
    """Only empty or unambiguously instrumental directives skip the singer."""
    if not lyrics or not lyrics.strip():
        return True
    meaningful = []
    for raw in lyrics.splitlines():
        line = raw.strip()
        if not line or line.startswith('//'):
            continue
        if line.casefold() in ('[end]', '[start]'):
            continue
        if line.startswith('[') and line.endswith(']'):
            value = line[1:-1].strip().casefold()
            if value in DIRECTIVES or value in ('intro', 'outro', 'interlude', 'solo', 'break', 'music break'):
                continue
            # A vocal heading without sung words still has no words to sing.
            if value in ('verse', 'chorus', 'bridge', 'pre-chorus', 'pre chorus'):
                continue
            # Unknown [tags] are structural, not permission to invent words.
            continue
        meaningful.append(re.sub(r'\s+', ' ', line).casefold())
    return not meaningful or all(x in DIRECTIVES for x in meaningful)


def seconds(abc: str) -> float:
    score = parse_abc(abc)
    return float(score.voices['Vocal'].time * 60 / score.bpm)


def _count_measures(text: str) -> int:
    """Count **expanded** native bars, including mixed `Z2|C...|` groups.

    The native parser expands Z2/Z3/Z4 separately for each bar; splitting on
    `|` alone undercounts mixed multi-measure rests, and previously rejected
    perfectly valid SheetSage2 ABC as two voices with different bar counts.
    The score is parsed with `parse_abc()` before this function is called, so
    the native parser has already verified 1..4 bars and equal paired timing.
    """
    from .engine import CHORD
    bars = [CHORD.sub('', x).strip() for x in text.split('|') if x.strip()]
    return sum(int(match.group(1) or '1') if (match := re.fullmatch(r'Z([2-4])?', bar)) else 1
               for bar in bars)



def _safe_rest_bar(units: int) -> str:
    """Represent an exact empty measure using YuE2 native rest durations."""
    from .vendor.yue2_cover_main.yue2_cover.abc_tools import DURATIONS
    chunks = []
    for size in sorted(DURATIONS, reverse=True):
        count, units = divmod(units, size)
        chunks += ['z' + str(size)] * count
    if units:
        raise ValueError('Cannot represent a chorded rest with native YuE2 ABC durations')
    return ''.join(chunks)


def _rest_staff_with_chords(score, bar_cursor: int, count: int) -> str:
    """Preserve exact chord change timestamps, even when vocal notes are muted."""
    from fractions import Fraction
    unit = score.unit * 4
    bars = score.voices['Vocal'].bars[bar_cursor:bar_cursor+count]
    events = score.voices['Vocal'].chords
    segments=[]
    if not any(start <= t < start + length for start,length,_ in bars for t,_ in events):
        return ('Z' + str(count) if count > 1 else 'Z') + '|'
    for start, duration, _meter in bars:
        finish = start + duration
        changes = sorted((t,c) for t,c in events if start <= t < finish)
        pointer = start
        result = ''
        for instant, chord in changes + [(finish, None)]:
            gap = (instant-pointer) / unit
            if gap.denominator != 1 or gap < 0:
                raise ValueError('Chord change not aligned to native ABC rest grid')
            if gap:
                result += _safe_rest_bar(int(gap))
            if chord is not None:
                result += '"' + chord + '"'
            pointer = instant
        # If there are no chords at time zero, the first segment already
        # emitted silence. Native parser enforces exact meter after emission.
        segments.append(result)
    return '|'.join(segments) + '|'

def instrumentalize_score(abc: str, *, max_duration: float = 0.0) -> tuple[str, dict]:
    """Keep Ins when present; otherwise promote the original Vocal melody to Ins.

    Both staffs are retained for the native composer, with no Vocal note events.
    Never merges simultaneous monophonic voices into unsupported ABC polyphony.
    """
    original = parse_abc(abc)
    lines = abc.splitlines()
    keys = sorted(original.music_lines)
    if len(keys) % 2 or any(original.music_lines[keys[i]] != 'Vocal' or
       original.music_lines[keys[i + 1]] != 'Ins' for i in range(0, len(keys), 2)):
        raise ValueError('Cannot safely create an instrumental cover: unsupported paired-voice layout')
    edits = []
    ins_preserved = 0
    melody_transferred = 0
    voiced_melody_muted = 0
    bar_cursor = 0
    for pos in range(0, len(keys), 2):
        vi, ii = keys[pos:pos + 2]
        count_v, count_i = _count_measures(lines[vi]), _count_measures(lines[ii])
        if count_v != count_i or count_v <= 0:
            raise ValueError(
                f'Instrumental group {pos // 2 + 1} has inconsistent expanded bar counts '
                f'(Vocal={count_v}, Ins={count_i}); input requires matched native voices'
            )
        vnotes = lines[vi]
        inotes = lines[ii]
        # Parse each token as actual note, not textual B within chord labels.
        # Detect using native score line-to-note membership via start/end bars.
        from .engine import CHORD, is_only_rests
        if not is_only_rests(vnotes) and is_only_rests(inotes):
            # Chord markers belong to Vocal in the native parser. Transfer only
            # pitched musical notes, not chord annotations, into Ins.
            lines[ii] = CHORD.sub('', vnotes)
            melody_transferred += 1
        else:
            ins_preserved += 1
            if not is_only_rests(vnotes):
                voiced_melody_muted += 1
        lines[vi] = _rest_staff_with_chords(original, bar_cursor, count_v)
        bar_cursor += count_v
        edits.append({'vocal_line': vi + 1, 'instrumental_line': ii + 1,
                      'bars': count_v, 'lead_moved': not is_only_rests(vnotes) and is_only_rests(inotes)})
    output = '\n'.join(lines) + ('\n' if abc.endswith('\n') else '')
    converted = parse_abc(output)
    if converted.voices['Vocal'].notes:
        raise ValueError('Instrumental conversion left Vocal note events')
    if converted.voices['Vocal'].chords != original.voices['Vocal'].chords:
        raise ValueError('Instrumental conversion changed harmonic chord events')
    if converted.voices['Vocal'].keys != original.voices['Vocal'].keys:
        raise ValueError('Instrumental conversion changed key-change chronology')
    if converted.voices['Vocal'].time != converted.voices['Ins'].time or \
       original.voices['Vocal'].time != converted.voices['Vocal'].time:
        raise ValueError('Instrumental conversion changed native score timing')
    trim = None
    if max_duration > 0 and seconds(output) > max_duration:
        output, trim = trim_instrumental_to_duration(output, max_duration)
    return output, {'decision': 'instrumental_cover', 'lyrics_mode': 'instrumental',
                    'vocal_note_events': 0, 'source_duration_s': round(seconds(abc), 3),
                    'output_duration_s': round(seconds(output), 3),
                    'max_duration_s': max_duration, 'instrumental_groups_preserved': ins_preserved,
                    'vocal_groups_transferred_to_ins': melody_transferred,
                    'vocal_groups_muted_over_existing_accompaniment': voiced_melody_muted,
                    'edits': edits, 'duration_fit': trim,
                    'audio_guarantee': 'native symbolic instrumental conditioning; listen or audit the generated audio'}


def trim_instrumental_to_duration(abc: str, max_duration: float, *, reserve_seconds: float = 3.0,
                                   max_tempo_boost: float = .08) -> tuple[str, dict]:
    if max_duration <= reserve_seconds:
        raise ValueError('max_duration too short for instrumental tail budget')
    score = parse_abc(abc)
    if seconds(abc) <= max_duration - reserve_seconds:
        return abc, {'action': 'not_required'}
    lines = abc.splitlines()
    ends = sorted(i for i, v in score.music_lines.items() if v == 'Ins')
    options = []
    for idx, end in enumerate(ends):
        prefix = lines[:end + 1]
        try:
            before = parse_abc('\n'.join(prefix) + '\n')
            length = float(before.voices['Vocal'].time * 60 / before.bpm)
            ratio = length / (max_duration - reserve_seconds)
            if ratio > 1 + max_tempo_boost:
                continue
            bpm = max(score.bpm, math.ceil(score.bpm * max(1, ratio)))
            if bpm / score.bpm > 1 + max_tempo_boost + 1e-9:
                continue
            tempo_idx = next(i for i, l in enumerate(prefix) if l.startswith('Q:'))
            prefix[tempo_idx] = f'Q:1/4={bpm}'
            attempt = '\n'.join(prefix) + '\n'
            parsed = parse_abc(attempt)
            if parsed.voices['Vocal'].notes or seconds(attempt) > max_duration - reserve_seconds + .001:
                continue
            options.append((idx + 1, attempt, bpm))
        except (ValueError, StopIteration, KeyError, IndexError):
            continue
    if not options:
        raise ValueError('Instrumental score cannot fit max_duration at a safe native-group boundary')
    groups, out, tempo = max(options, key=lambda x: x[0])
    return out, {'action': 'trim_whole_groups', 'groups_removed': len(ends) - groups,
                 'bpm_before': score.bpm, 'bpm_after': tempo,
                 'output_duration_s': round(seconds(out), 3), 'reserved_tail_s': reserve_seconds}

INSTRUMENTAL_SUFFIX = ('Instrumental performance only. No singing, no spoken voice, '
                       'no backing vocals, no vocalizations, no scat, no humming. '
                       'Let musical instruments carry the melody and harmony.')


def instrumental_aware_style(style: str, lyrics: str) -> str:
    """Do not send conflicting singer instructions into an instrumental render."""
    if not instrumental_requested(lyrics):
        return style
    return str(style).strip() + ('\n\n' if str(style).strip() else '') + INSTRUMENTAL_SUFFIX


class YuE2InstrumentalAwareStyle:
    """The Standard graph passes the exact same Style unless Lyrics are empty."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'style': ('STRING', {'forceInput': True}),
            'lyrics': ('STRING', {'forceInput': True}),
        }}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('effective_style',)
    FUNCTION = 'prepare'
    CATEGORY = 'audio/YuE2'

    def prepare(self, style, lyrics):
        return (instrumental_aware_style(style, lyrics),)

NODE_CLASS_MAPPINGS={'YuE2InstrumentalAwareStyle': YuE2InstrumentalAwareStyle}
NODE_DISPLAY_NAME_MAPPINGS={'YuE2InstrumentalAwareStyle': 'YuE2 · Instrumental-Aware Style'}
