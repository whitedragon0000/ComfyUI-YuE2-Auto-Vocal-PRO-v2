"""ABC Vocal Router v2, built for ComfyUI YuE2 and SheetSage2.

This module is pure Python: no ComfyUI or third-party dependencies.
Scope: native YuE2 two-voice ABC dialect (Vocal/Ins), not a full ABC interpreter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

SECTION_COMMENT = re.compile(r'^\s*%\s*(.+?)\s*$')
SECTION_BRACKET = re.compile(r'^\s*\[\s*([^\]]+)\s*\]\s*$')
VOICE_BODY = re.compile(r'^\s*V:\s*(Vocal|Ins)\s*$', re.I)
CHORD = re.compile(r'"[^"\n]*"')
BODY_META = re.compile(r'^\s*([MLKQ]):\s*(.+?)\s*$')
INLINE_META = re.compile(r'\[(?:[KMLQ]:)[^\]]*\]')
NOTE = re.compile(r"(?:[\^_=]{1,2})?[A-Ga-g][,']*(?:\d+(?:/\d+)?|/+)?")
REST = re.compile(r'(?:[zZxX]\d*(?:/\d+)?|[zZxX]/+)')

DEFAULT_INSTRUMENTAL = ('instrumental','instrumental break','intro','interlude','intermezzo',
    'solo','guitar solo','keyboard solo','drum solo','synth solo','sax solo',
    'music break','breakdown','outro','ending','instrumental outro','opening',
    'solo strumentale','strumentale','assolo','introduzione','finale strumentale')
DEFAULT_VOCAL = ('verse','strofa','chorus','ritornello','refrain','pre chorus','prechorus',
    'post chorus','bridge','ponte','hook','drop','vocal','duet','rap','spoken',
    'build','climax','final chorus','coda vocale')


def simplify_tag(tag: str) -> str:
    # Tolerates user-defined Suno-style [Verse 2 - female vocal] etc.
    tag = re.sub(r'[_\-]+', ' ', tag.casefold())
    return re.sub(r'\s+', ' ', tag).strip()


def parse_tag_field(text: str) -> tuple[str, ...]:
    return tuple(simplify_tag(x) for x in re.split(r'[,;\n]+', text or '') if x.strip())


def tag_matches(tag: str, patterns: tuple[str,...]) -> bool:
    norm = simplify_tag(tag)
    # Matches phrase as words, e.g. 'verse 2' matches 'verse', not 'converse'.
    return any(re.search(r'(?<!\w)'+re.escape(p)+r'(?!\w)',norm) for p in patterns if p)


def lyrics_sections(lyrics: str) -> list[str]:
    return [m.group(1).strip() for line in lyrics.splitlines()
            if (m := SECTION_BRACKET.match(line))]


def lyrics_has_words(lyrics: str) -> bool:
    return any(line.strip() and not SECTION_BRACKET.fullmatch(line.strip())
               and not line.strip().startswith('//') for line in lyrics.splitlines())


def is_note_content(score: str) -> bool:
    clean = CHORD.sub('', INLINE_META.sub('', score))
    clean = re.sub(r'![^!]*!', '', clean)  # ABC decorations
    return NOTE.search(clean) is not None


def is_only_rests(score: str) -> bool:
    clean = CHORD.sub('', INLINE_META.sub('', score))
    clean = re.sub(r'![^!]*!', '', clean)
    clean = re.sub(r'\s|\||\[|\]', '', clean)
    clean = REST.sub('',clean)
    return not clean


def split_bars(text: str) -> list[str]:
    # Flat scores as produced by SheetSage2; preserve note and chord tokens.
    return [part.strip() for part in text.replace('\n','').split('|') if part.strip()]



def _unit_duration(token: str) -> Fraction:
    """ABC note/rest length in L units. Supports integers, slash and slash fractions."""
    match=re.search(r'(\d+(?:/\d+)?|/+)\Z',token)
    if match:
        value=match.group(1)
        if value.startswith('/'):
            return Fraction(1,2**len(value))
        return Fraction(value)
    return Fraction(1)


def timed_chords(vocal_bar: str) -> list[tuple[Fraction,str]]:
    """Track chord changes relative to beats, even when vocal staff only holds rests."""
    token = re.compile(r'"[^"\n]*"|(?:[\^_=]{1,2})?[A-Ga-g][,\']*(?:\d+(?:/\d+)?|/+)?|[zZxX](?:\d+(?:/\d+)?|/+)')
    position=Fraction(0)
    output=[]
    for m in token.finditer(vocal_bar):
        t=m.group(0)
        if t.startswith('"'):output.append((position,t))
        else:position+=_unit_duration(t)
    return output


def weave_chords(lead: str, timed: list[tuple[Fraction,str]]) -> str:
    """Put chords before first instrumental note reaching each time offset.

    If chord falls inside a held note, place before next onset rather than
    splitting the note/tie; the original note content is never rewritten.
    """
    if not timed:return lead
    scan=re.compile(r'(?:[\^_=]{1,2})?[A-Ga-g][,\']*(?:\d+(?:/\d+)?|/+)?|[zZxX](?:\d+(?:/\d+)?|/+)')
    position=Fraction(0);last=0; out=[];idx=0
    for m in scan.finditer(lead):
        out.append(lead[last:m.start()])
        while idx<len(timed) and timed[idx][0] <= position:
            out.append(timed[idx][1]);idx+=1
        out.append(m.group());position+=_unit_duration(m.group());last=m.end()
    out.append(lead[last:])
    while idx<len(timed):out.append(timed[idx][1]);idx+=1
    return ''.join(out)

def extract_music(lines: list[str]) -> tuple[list[str],str]:
    meta=[]; music=[]
    for line in lines:
        if BODY_META.match(line):meta.append(line)
        elif line.strip() and not line.lstrip().startswith('%'):music.append(line.strip())
    return meta, ''.join(music)


def measure_rests(meter: str, default_length: str, count: int) -> str:
    """Represent 1–4 measures of silence in YuE2's native two-voice ABC dialect.

    `Z4|` is a four-MEASURE rest. A single measure is `Z|` (not Z1|). This is the representation used in
    the previously verified hand-corrected cover score, unlike four separate
    `z16|` bars. Keep the multi-measure shorthand to preserve section phrases.
    Meter/length are intentionally not used: uppercase Z counts full measures.
    """
    if not 1 <= count <= 4:
        raise ValueError('Native YuE2 groups support 1–4 measures; split larger groups before encoding rests')
    # Z1 and Z5+ are invalid; only one Z / Z2 / Z3 / Z4 per native group.
    return ('Z' + (str(count) if count > 1 else '') + '|')


@dataclass
class Report:
    groups: int = 0
    converted_groups: int = 0
    converted_bars: int = 0
    already_vocal: int = 0
    instrumental_groups: int = 0
    expanded_placeholders: int = 0
    skipped: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    lyric_tags: list[str] = field(default_factory=list)
    def render(self) -> str:
        s = (f'YuE2 Auto Vocal PRO V2 | groups={self.groups}, converted_groups={self.converted_groups}, '
             f'converted_bars={self.converted_bars}, existing_vocal_groups={self.already_vocal}, '
             f'instrumental_groups={self.instrumental_groups}, expanded_placeholders={self.expanded_placeholders}.')
        if self.sections:s += '\nABC sections: '+', '.join(dict.fromkeys(self.sections))
        if self.lyric_tags:s += '\nLyrics tags (preserved verbatim): '+', '.join(self.lyric_tags)
        if self.skipped:s += '\nWarnings:\n- ' + '\n- '.join(self.skipped)
        return s


def classify(section: str, policy: str, vocal_extra: tuple[str,...],
             instrumental_extra: tuple[str,...]) -> bool:
    # Explicit user overrides take priority over built-in tag classifications.
    if tag_matches(section,instrumental_extra):return False
    if tag_matches(section,vocal_extra):return True
    if policy == 'all_sections':return True
    if tag_matches(section,DEFAULT_INSTRUMENTAL):return False
    if policy == 'verses_and_choruses':
        return tag_matches(section, ('verse','chorus','strofa','ritornello','refrain'))
    # all_of_rest: any tag (including user-created ones) except explicitly instrumental
    return True


def convert_abc(abc: str, lyrics: str = '', *, section_policy: str = 'all_of_rest',
        instrumental_handling: str = 'replace_with_rests',
        instrumental_tags: str = '', vocal_tags: str = '',
        existing_vocal: str = 'preserve', strict: bool = True) -> tuple[str,str,str]:
    """Route Ins lead to Vocal, correcting SheetSage2's short Z4 placeholders.

    Returns (ABC, unchanged lyrics, human-readable diagnostic report).
    The exact Lyrics string is passed through; no invented lyrics or tag rewriting.
    """
    if section_policy not in ('all_of_rest','verses_and_choruses','all_sections'):
        raise ValueError('Invalid section_policy: '+section_policy)
    if instrumental_handling not in ('replace_with_rests','keep_original'):
        raise ValueError('Invalid instrumental_handling: '+instrumental_handling)
    if existing_vocal not in ('preserve','overwrite'):
        raise ValueError('Invalid existing_vocal: '+existing_vocal)
    if not abc or not abc.strip():
        raise ValueError('ABC input is empty. Connect SheetSage2 to Auto Vocal PRO.')
    inst = parse_tag_field(instrumental_tags)
    voc = parse_tag_field(vocal_tags)
    report=Report(lyric_tags=lyrics_sections(lyrics))
    if not lyrics_has_words(lyrics):
        report.skipped.append('No lyric words found: an instrumental result may be expected.')
        if strict:raise ValueError('Lyrics are empty or contain only section tags. Add sung lyrics.')
    # Native parsing is authoritative for bar count and event chronology.
    # SheetSage compresses multi-measure rests as Z2/Z3/Z4, which are NOT
    # equivalent to the number of textual `|` separators.
    from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc, AbcError
    from .instrumental_mode import _count_measures
    try:
        native_before = parse_abc(abc)
    except AbcError:
        # Keep the original conversion error behavior for unsupported inputs;
        # downstream native validation must still reject unparseable scores.
        native_before = None
    lines=abc.splitlines()
    # Decide cross-group tie regions atomically before altering any staff.
    # The original source score is preserved when the whole tied phrase
    # cannot be converted safely; no emergency tie deletion or note rewrites.
    from .tie_group_contract import blocked_tie_groups
    tie_blocked = blocked_tie_groups(lines, section_policy, inst, voc, existing_vocal)
    # Header declarations contain additional arguments; body voice markers are exact V: Vocal / V: Ins.
    out=[]; i=0; section=''; meter='4/4'; length='1/16';
    for line in lines:
        m=BODY_META.match(line)
        if m and m.group(1)=='M':meter=m.group(2)
        if m and m.group(1)=='L':length=m.group(2)
        if line.strip().startswith('K:'):break
    while i<len(lines):
        line=lines[i]
        m=SECTION_COMMENT.match(line)
        if m:
            section=m.group(1).strip()
            report.sections.append(section)
            out.append(line);i+=1;continue
        # ABC structure is authoritative; arbitrary Suno lyrics tags are retained downstream.
        v=VOICE_BODY.match(line)
        if not v or v.group(1).casefold()!='vocal':
            out.append(line); i+=1
            if (m:=BODY_META.match(line)):
                if m.group(1)=='M':meter=m.group(2)
                if m.group(1)=='L':length=m.group(2)
            continue
        j=i+1
        while j<len(lines) and not VOICE_BODY.match(lines[j]) and not SECTION_COMMENT.match(lines[j]):j+=1
        if j>=len(lines) or not (w:=VOICE_BODY.match(lines[j])) or w.group(1).casefold()!='ins':
            out.extend(lines[i:j]);i=j;continue
        k=j+1
        while k<len(lines) and not VOICE_BODY.match(lines[k]) and not SECTION_COMMENT.match(lines[k]):k+=1
        voc_meta,voc_score=extract_music(lines[i+1:j]); ins_meta,ins_score=extract_music(lines[j+1:k])
        if not (voc_score and ins_score):
            report.skipped.append(f'Section {section!r}: missing Vocal/Ins music in group near line {i+1}.')
            out.extend(lines[i:k]);i=k;continue
        report.groups+=1
        allowed=classify(section,section_policy,voc,inst)
        if i in tie_blocked:
            report.skipped.append(f'Section {section!r}, line {i+1}: preserved original cross-group tied phrase atomically.')
            out.extend(lines[i:k]);i=k;continue
        if not allowed:
            report.instrumental_groups+=1
            out.extend(lines[i:k]);i=k;continue
        vocal_bars=split_bars(voc_score); ins_bars=split_bars(ins_score)
        if not ins_bars:
            out.extend(lines[i:k]);i=k;continue
        # IMPORTANT: a native Z4| spans four bars, even though it has one pipe.
        # Detect real vocal notes BEFORE comparing textual bar separators.
        # This also makes converted native ABC safe to pass through again.
        if existing_vocal=='preserve' and any(is_note_content(x) for x in vocal_bars):
            report.already_vocal+=1
            out.extend(lines[i:k]);i=k;continue
        vocal_expanded = _count_measures(voc_score)
        ins_expanded = _count_measures(ins_score)
        if native_before is not None and vocal_expanded != ins_expanded:
            raise ValueError(f'Score group at line {i+1} has unequal expanded measures: '
                             f'Vocal={vocal_expanded}, Ins={ins_expanded}')
        if not any(is_note_content(b) for b in ins_bars):
            # `overwrite` does not mean deleting an existing singer when the
            # instrumental staff has nothing to replace it with. This is a
            # normal YuE2 score, not a failed conversion.
            if any(is_note_content(b) for b in vocal_bars):
                report.already_vocal += 1
            report.skipped.append(f'Section {section!r}, line {i+1}: no instrumental notes to promote.')
            out.extend(lines[i:k]);i=k;continue
        # Preserve every musical beat, not the textual number of bars.
        # Transfer the complete instrumental staff if native rests are mixed
        # (e.g. Vocal Z2|Z| and Ins B16|Z2|). Literal pipe-to-pipe
        # pairing would change the duration and corrupt the ABC.
        complex_rests = (len(vocal_bars) != len(ins_bars) or
                         any(re.fullmatch(r'Z[2-4]', CHORD.sub('', x).strip()) for x in ins_bars))
        if complex_rests and not any(CHORD.search(x) for x in vocal_bars):
            transferred = CHORD.sub('', ins_score)
            out.append(lines[i]); out.extend(voc_meta); out.append(transferred.rstrip('|')+'|')
            out.append(lines[j]); out.extend(ins_meta)
            if instrumental_handling == 'keep_original':
                out.append(ins_score.rstrip('|')+'|')
            else:
                out.append(measure_rests(meter,length,ins_expanded))
            report.converted_groups += 1
            report.converted_bars += ins_expanded
            report.expanded_placeholders += 1
            i=k
            continue
        if len(vocal_bars)!=len(ins_bars):
            # Chorded mixed rest groups need exact event reconstruction; no
            # silent approximate chord shifts are allowed.
            report.skipped.append(f'Section {section!r}, line {i+1}: complex chorded rest layout; preserved')
            out.extend(lines[i:k]);i=k;continue
        composed=[]
        for vb, ib in zip(vocal_bars,ins_bars):
            # Chords are stored on the vocal staff. If absent, preserve chords from Ins.
            chord_positions=timed_chords(vb)
            ib_notes=CHORD.sub('',ib).strip()
            if not chord_positions: chord_positions=timed_chords(ib)
            # Keep chord changes near their original offsets, not all at bar onset.
            composed.append(weave_chords(ib_notes,chord_positions))
        # Do not change meter/key headers; replicate in the target voice, preserve file format.
        out.append(lines[i]);out.extend(voc_meta);out.append('|'.join(composed)+'|')
        out.append(lines[j]);out.extend(ins_meta)
        if instrumental_handling=='keep_original':
            out.append(ins_score if ins_score.endswith('|') else ins_score+'|')
        else:
            local_m=next((m.group(2) for x in reversed(voc_meta+ins_meta) if (m:=BODY_META.match(x)) and m.group(1)=='M'),meter)
            local_l=next((m.group(2) for x in reversed(voc_meta+ins_meta) if (m:=BODY_META.match(x)) and m.group(1)=='L'),length)
            out.append(measure_rests(local_m,local_l,ins_expanded))
        report.converted_groups+=1;report.converted_bars+=ins_expanded
        for x in voc_meta+ins_meta:
            if (m:=BODY_META.match(x)):
                if m.group(1)=='M':meter=m.group(2)
                if m.group(1)=='L':length=m.group(2)
        i=k
    if (report.converted_groups == 0 and report.already_vocal == 0 and
        not (native_before is not None and native_before.voices['Vocal'].notes)):
        msg='ZERO vocal conversions. No eligible melodic Ins/Vocal groups: see ABC sections, already-empty source, and section policy.'
        report.skipped.append(msg)
        if strict:raise ValueError(report.render())
    # Invariant: every native before-score that was valid must remain valid.
    # This comparison prevents wrong bar counts, key/chord changes or time drift.
    produced = '\n'.join(out).rstrip()+'\n'
    if native_before is not None:
        try:
            native_after = parse_abc(produced)
            for staff in ('Vocal', 'Ins'):
                if native_after.voices[staff].bars != native_before.voices[staff].bars:
                    raise ValueError(f'ABC conversion changed {staff} bar/meter grid')
                if native_after.voices[staff].keys != native_before.voices[staff].keys:
                    raise ValueError(f'ABC conversion changed {staff} key chronology')
            if native_after.voices['Vocal'].chords != native_before.voices['Vocal'].chords:
                raise ValueError('ABC conversion changed harmonic chord chronology')
        except (AbcError, ValueError) as exc:
            # Transactional guarantee: an already-valid source MUST NOT turn
            # into an invalid score. The original musical evidence is the
            # only defensible fallback, not invented notes or deleted ties.
            # The full reason is surfaced in the conversion report.
            report.skipped.append('ABC conversion rolled back atomically: '
                                  f'{type(exc).__name__}: {exc}; original score preserved.')
            report.converted_groups = 0
            report.converted_bars = 0
            report.expanded_placeholders = 0
            return abc, lyrics, report.render()
    return produced,lyrics,report.render()
