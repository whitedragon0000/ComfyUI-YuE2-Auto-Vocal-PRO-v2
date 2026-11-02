"""Conservative section and lyrics-coverage analysis for YuE2 two-voice ABC.

No phoneme-to-note timing is claimed. Only modifies section labels when 1:1
mapping is unambiguous. Lyrics are never extended unless repeat_existing
is explicitly selected by the user.
Pure standard-library Python.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .engine import BODY_META, CHORD, DEFAULT_INSTRUMENTAL, NOTE, SECTION_BRACKET, SECTION_COMMENT, VOICE_BODY, is_note_content, simplify_tag, split_bars, tag_matches

TOKEN = re.compile(r"(?:[\^_=]{1,2})?[A-Ga-g][,']*(?:\d+(?:/\d+)?|/+)?")
VOCAL_LINES = re.compile(r"[^\W\d_]+", re.UNICODE)
VOWEL_GROUPS = re.compile(r"[aeiouàáèéìíòóùú]+", re.I)
CANONICAL = (
    ('pre chorus', 'pre chorus'), ('prechorus', 'pre chorus'),
    ('post chorus', 'post chorus'), ('postchorus', 'post chorus'),
    ('final chorus', 'chorus'), ('chorus', 'chorus'), ('ritornello', 'chorus'),
    ('verse', 'verse'), ('strofa', 'verse'), ('bridge', 'bridge'), ('ponte', 'bridge'),
    ('refrain', 'chorus'), ('hook', 'hook'), ('rap', 'rap'),
    ('spoken', 'spoken'), ('outro', 'outro'), ('intro', 'intro'),
)

@dataclass
class ScoreRun:
    heading_index: int | None
    label: str
    bars: int = 0
    onsets: int = 0

@dataclass
class LyricPart:
    tag: str | None
    tag_line: int | None
    start: int
    end: int
    lines: list[str] = field(default_factory=list)
    @property
    def sung_lines(self) -> list[str]:
        return [s.strip() for s in self.lines if s.strip() and not s.lstrip().startswith('//')]


def _strip_abc_fragments(s: str) -> str:
    s = CHORD.sub('', s)
    s = re.sub(r'\[(?:[KMLQ]:)[^\]]*\]', '', s)
    s = re.sub(r'![^!]*!', '', s)
    return s


def score_runs(abc: str) -> list[ScoreRun]:
    """Collect vocal sections, including grouped 4-bar SheetSage2 passages."""
    lines = abc.splitlines()
    runs: list[ScoreRun] = []
    run: ScoreRun | None = None
    heading_idx = None
    heading = ''
    voice = None
    in_header = True
    for idx, line in enumerate(lines):
        if in_header and line.strip().startswith('K:'):
            in_header = False
            continue
        if in_header:
            continue
        if m := SECTION_COMMENT.match(line):
            heading_idx = idx
            heading = m.group(1).strip()
            run = ScoreRun(heading_idx, heading)
            runs.append(run)
            voice = None
            continue
        if m := VOICE_BODY.match(line):
            voice = m.group(1).casefold()
            continue
        if voice != 'vocal' or not line.strip() or BODY_META.match(line) or line.lstrip().startswith('%'):
            continue
        if not is_note_content(line):
            continue
        if run is None:
            run = ScoreRun(None, '<unlabeled>')
            runs.append(run)
        bars = split_bars(line)
        run.bars += len(bars)
        run.onsets += len(TOKEN.findall(_strip_abc_fragments(line)))
    return [r for r in runs if r.bars]


def lyric_parts(lyrics: str) -> tuple[list[str], list[LyricPart]]:
    lines = lyrics.splitlines()
    parts: list[LyricPart] = []
    current = LyricPart(None, None, 0, len(lines))
    for idx, line in enumerate(lines):
        m = SECTION_BRACKET.match(line)
        if m:
            if current.lines or current.tag_line is not None:
                current.end = idx
                parts.append(current)
            current = LyricPart(m.group(1).strip(), idx, idx + 1, len(lines))
        else:
            current.lines.append(line)
    current.end = len(lines)
    if current.lines or current.tag_line is not None:
        parts.append(current)
    return lines, parts


def _canonical(tag: str) -> str:
    norm = simplify_tag(tag)
    for pattern, name in CANONICAL:
        if tag_matches(norm, (pattern,)):
            return name
    return norm.split(' - ', 1)[0].strip() or 'verse'


def _syllable_estimate(lines: list[str]) -> int:
    # Italian-friendly, language-agnostic rough estimate; not syllabification.
    return sum(max(1, len(VOWEL_GROUPS.findall(word)))
               for line in lines for word in VOCAL_LINES.findall(line))


def align_lyrics(abc: str, lyrics: str, *, mode: str = 'match_sections') -> tuple[str, str, str]:
    if mode not in ('report_only', 'match_sections', 'repeat_existing'):
        raise ValueError('Unknown lyrics_alignment mode: ' + mode)
    runs = score_runs(abc)
    abc_lines = abc.splitlines()
    lyric_lines, all_parts = lyric_parts(lyrics)
    parts = [p for p in all_parts if p.sung_lines]
    diagnostics = [f'Lyrics alignment: mode={mode}; vocal_sections={len(runs)}; lyric_sections={len(parts)}.']
    if not runs:
        diagnostics.append('WARNING: ABC has no vocal note sections.')
        return abc, lyrics, '\n'.join(diagnostics)
    if not parts:
        diagnostics.append('WARNING: No sung lyric lines to align; enter Lyrics before generation.')
        return abc, lyrics, '\n'.join(diagnostics)
    compatible = len(runs) == len(parts)
    if not compatible:
        diagnostics.append(f'WARNING: {len(runs)} ABC vocal sections vs {len(parts)} lyric sections with words; no automatic section relabeling/repetition. Split or combine your Lyrics sections manually.')
    if compatible and mode in ('match_sections', 'repeat_existing'):
        for run, part in zip(runs, parts):
            if run.heading_index is not None and part.tag is not None:
                label = _canonical(part.tag)
                if label and simplify_tag(run.label) != label:
                    abc_lines[run.heading_index] = '% ' + label
                    diagnostics.append(f'Section label: %{run.label} -> %{label} (lyrics tag [{part.tag}]).')
            elif part.tag is None:
                diagnostics.append('WARNING: Lyrics have no [Section] headings; ABC section label preserved.')
    extras: dict[int,list[str]] = {}
    for index, run in enumerate(runs):
        if not compatible:
            break
        part = parts[index]
        original = part.sung_lines
        estimated_syllables = _syllable_estimate(original)
        bpl = run.bars / len(original)
        onsets_per_syllable = (run.onsets / max(1, estimated_syllables))
        diagnostics.append(
            f'Section {index+1}: abc={run.label}, lyrics={part.tag or "(unlabeled)"}, '
            f'bars={run.bars}, note_onsets={run.onsets}, lyric_lines={len(original)}, '
            f'estimated_syllables={estimated_syllables}, bars_per_line={bpl:.2f}, '
            f'notes_per_estimated_syllable={onsets_per_syllable:.2f}.')
        if bpl > 3:
            diagnostics.append(f'WARNING section {index+1}: sparse lyrics ({bpl:.1f} bars/line); non-lexical vocals or repetition are possible.')
        if mode == 'repeat_existing' and bpl > 2.5:
            # Opt-in ONLY: preserve each existing line and repeat those exact lines.
            # Limiting repeats avoids runaway tokens and accidental lyric invention.
            wanted = max(len(original), math.ceil(run.bars / 2))
            times = min(8, math.ceil(wanted / len(original)))
            extra = original * (times - 1)
            extras[part.end] = extras.get(part.end, []) + extra
            diagnostics.append(f'Section {index+1}: explicitly repeated existing lines {times}x; added_lines={len(extra)}.')
    if mode == 'repeat_existing' and compatible and extras:
        produced=[]
        for idx,line in enumerate(lyric_lines):
            if idx in extras:
                produced.extend(extras[idx])
            produced.append(line)
        if len(lyric_lines) in extras:
            produced.extend(extras[len(lyric_lines)])
        # Preserve trailing newline if the input had one.
        lyrics_out = '\n'.join(produced) + ('\n' if lyrics.endswith('\n') else '')
    else:
        lyrics_out = lyrics
    return ('\n'.join(abc_lines).rstrip()+'\n' if mode != 'report_only' else abc), lyrics_out, '\n'.join(diagnostics)
