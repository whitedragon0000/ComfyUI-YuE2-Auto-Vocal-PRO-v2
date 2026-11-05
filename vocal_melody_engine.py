"""Bounded symbolic lyric-to-score optimizer for native YuE2 ABC.

This module does not claim that native YuE2 obeys note/phoneme constraints.
It returns an auditable *plan* and an ABC that is checked by the native
score parser. Never substitutes/reference-transcribes target lyrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import json
import math
import re
import unicodedata

from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc, TOKEN as ABC_TOKEN, DURATIONS
from .alignment import lyric_parts
from .lyric_contract import automatic_lyrics, _tokens

MODES = ('Adaptive', 'Music adapts to Lyrics', 'Lyrics adapt to Music')
VOWELS = re.compile(r'[aeiouàáèéìíòóùú]+', re.IGNORECASE)
WORDS = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
END_TAG = re.compile(r'^\s*\[(?:End)\]\s*$', re.I)


class AlignmentPlanningError(ValueError):
    """A requested symbolic change cannot be made without violating invariants."""


@dataclass(frozen=True)
class Syllable:
    word: str
    word_index: int
    index: int
    word_syllables: int
    stressed: bool
    ends_word: bool
    ends_phrase: bool


@dataclass(frozen=True)
class Section:
    label: str
    first_bar: int
    last_bar: int
    notes: tuple
    seconds: float


@lru_cache(maxsize=4096)
def _syl_count(word: str) -> int:
    # Opt-in higher quality Italian syllabification if pyphen is installed.
    # Keep the heuristic deterministic and dependency-free otherwise.
    lexical = re.sub(r"[^\wàáèéìíòóùú]", '', word, flags=re.I)
    try:
        import pyphen
        hyph = pyphen.Pyphen(lang='it_IT')
        parts = [x for x in hyph.inserted(lexical).split('-') if x]
        if len(parts) > 1:
            return len(parts)
    except (ImportError, KeyError, ValueError):
        pass
    return max(1, len(VOWELS.findall(lexical)))


def syllabify(lyrics: str) -> list[Syllable]:
    """Return ordered estimated syllable slots; never changes the lyrics."""
    _, parts = lyric_parts(lyrics)
    lines = [line for p in parts for line in p.sung_lines]
    syllables = []
    word_index = 0
    for line in lines:
        found = list(WORDS.finditer(line))
        for i, match in enumerate(found):
            word = match.group(0)
            n = _syl_count(word)
            accent = re.search('[àáèéìíòóùú]', word, flags=re.I)
            stressed_index = min(n - 1, n - 2 if n > 1 else 0)
            if accent:
                before = word[:accent.start() + 1]
                stressed_index = min(n-1, max(0, len(VOWELS.findall(before))-1))
            punctuation = line[match.end():found[i+1].start() if i+1 < len(found) else len(line)]
            ends_phrase = bool(re.search(r'[.!?;:,]', punctuation)) or i == len(found)-1
            for si in range(n):
                syllables.append(Syllable(word, word_index, si, n,
                                          si == stressed_index,
                                          si == n-1, si == n-1 and ends_phrase))
            word_index += 1
    return syllables


def _expanded_bar_count(music: str) -> int:
    count = 0
    for bar in music.rstrip('|').split('|'):
        bar = bar.strip()
        m = re.fullmatch(r'Z([2-4])', bar)
        count += int(m.group(1)) if m else 1
    return count


def score_sections(abc: str) -> list[Section]:
    """Native parser owns times; section headings own lyric grouping."""
    score = parse_abc(abc)
    lines = abc.splitlines()
    vocal_lines = sorted(idx for idx, name in score.music_lines.items() if name == 'Vocal')
    bars = score.voices['Vocal'].bars
    notes = score.voices['Vocal'].notes
    groups = []
    cursor = 0
    label = 'Verse'
    previous_line = 7
    group_idx = -1
    for idx in vocal_lines:
        heading = [line[2:].strip() for line in lines[previous_line+1:idx] if line.startswith('% ')]
        if heading: label = heading[-1]
        count = _expanded_bar_count(lines[idx])
        group_bars = bars[cursor:cursor+count]
        if len(group_bars) != count:
            raise AlignmentPlanningError('ABC vocal bar count differs from native score parser')
        start, end = group_bars[0][0], group_bars[-1][0] + group_bars[-1][1]
        group_notes = tuple((t, pitch, duration) for t,pitch,duration in notes if start <= t < end)
        new_heading = bool(heading) or not groups
        if new_heading:
            group_idx += 1
            groups.append({'label': label, 'first': cursor, 'last': cursor+count, 'notes': list(group_notes)})
        else:
            groups[-1]['last'] = cursor+count
            groups[-1]['notes'].extend(group_notes)
        cursor += count
        previous_line = idx
    if cursor != len(bars):
        raise AlignmentPlanningError('Music groups and parsed vocal bars disagree')
    result = []
    for group in groups:
        if not group['notes']:
            continue
        first = bars[group['first']][0]
        last = bars[group['last']-1][0] + bars[group['last']-1][1]
        result.append(Section(group['label'], group['first'], group['last'],
                              tuple(group['notes']), float((last-first)*60/score.bpm)))
    return result


def _sections_lyrics(abc: str, lyrics: str, sections: list[Section]) -> tuple[str, list[str]]:
    planned, info = automatic_lyrics(abc, lyrics)
    if info['status'] != 'planned':
        raise AlignmentPlanningError('Cannot allocate all target words to vocal score sections: '+info['reason'])
    _, parts = lyric_parts(planned)
    # Native blocks include instrumental sections and may include vocal
    # sections with no assigned words. Align by the actual music section grid,
    # NOT by filtering out empty text and shifting all following words.
    from .section_contract import native_sections
    section_grid = native_sections(abc)
    if len(parts) != len(section_grid):
        raise AlignmentPlanningError('Unmatched native lyric and score section counts after allocation')
    groups = ['\n'.join(part.sung_lines)
              for part,section in zip(parts,section_grid) if section.vocal_onsets]
    if len(groups) != len(sections):
        raise AlignmentPlanningError('Native vocal section grid cannot be matched to lyric slots')
    original, _ = _tokens(lyrics)
    output, _ = _tokens(planned)
    if original != output:
        raise AlignmentPlanningError('User word sequence changed during lyric allocation')
    return planned, groups


def _align_syllables(notes: tuple, syls: list[Syllable], bpm: int) -> tuple[list[dict], dict]:
    """Monotonic bounded DP; segment durations and strong beats shape cost."""
    count = len(notes)
    if not count or not syls:
        return [], {'cost':0.0,'syllables':len(syls),'notes':count,'status':'empty'}
    times = [float(n[2])*60/bpm for n in notes]
    if len(syls) > count:
        # Score is immutable in Lyrics-adapt mode. Multiple slots per note are
        # a PLAN ONLY, not acoustic sub-note timing enforced by YuE2.
        allocated = []
        for si, syl in enumerate(syls):
            ni = min(count-1, int(si*count/len(syls)))
            t,p,d = notes[ni]
            allocated.append({'word':syl.word, 'word_index':syl.word_index,
                              'syllable_index':syl.index, 'note_index':ni,
                              'note_start_s':round(float(t)*60/bpm,4),
                              'note_duration_s':round(float(d)*60/bpm,4),
                              'stressed':syl.stressed,'shared_note':True})
        return allocated, {'cost':round(2.8*(len(syls)-count),3),
                           'syllables':len(syls),'notes':count,'status':'shared_note_warning'}
    n = len(syls)
    # Exact DP is bounded. Large full songs use monotonic weighted partitions
    # with explicit provenance, instead of exhausting ComfyUI CPU/RAM.
    if n > 280 or count > 360 or n * count > 45000 or count > n * 8:
        allocated = []
        last = 0
        for si,syl in enumerate(syls):
            wanted = round((si+1)*count/n)
            stop = min(count-(n-si-1), max(last+1,wanted))
            t,p,d = notes[last]
            allocated.append({'word':syl.word,'word_index':syl.word_index,
                              'syllable_index':syl.index,'note_index':last,
                              'note_count':stop-last,'note_start_s':round(float(t)*60/bpm,4),
                              'note_duration_s':round(sum(times[last:stop]),4),
                              'stressed':syl.stressed,'melisma':stop-last>1})
            last=stop
        return allocated, {'cost':round((count-n)*.26,3),'syllables':n,
                           'notes':count,'status':('underconstrained_note_density_warning' if count > n * 8 else 'bounded_monotonic_fallback')}
    mean = sum(times)/n
    weights = [1.24 if syl.stressed else (1.13 if syl.ends_phrase else 1.0) for syl in syls]
    total_weights = sum(weights)
    maximum = min(8, count-n+1)

    @lru_cache(maxsize=None)
    def dp(si: int, ni: int):
        if si == n:
            return (0.0, ()) if ni == count else (math.inf, ())
        best = (math.inf, ())
        # Reserve at least one note for each remaining syllable.
        max_step = min(maximum, count - ni - (n-si-1))
        for step in range(1, max_step+1):
            assigned = sum(times[ni:ni+step])
            desired = sum(times)*weights[si]/total_weights
            dur_penalty = 0.65*(math.log(max(0.03,assigned)/max(0.03,desired))**2)
            melisma_penalty = 0.23*(step-1)**2
            stress_penalty = 0.11 if syls[si].stressed and float(notes[ni][0])%4 > .000001 else 0.0
            # A rest inside a sung word is worse than a phrase-boundary breath.
            word_break_penalty = 0.0
            if si and syls[si].word_index == syls[si-1].word_index and ni:
                pause = float(notes[ni][0] - notes[ni-1][0] - notes[ni-1][2]) * 60/bpm
                word_break_penalty = max(0.0,pause) * 3.0
            end_penalty = 0.12 if syls[si].ends_phrase and assigned < mean*.65 else 0.0
            cost, continuation = dp(si+1, ni+step)
            trial = (dur_penalty+melisma_penalty+stress_penalty+word_break_penalty+end_penalty+cost, (step,)+continuation)
            if trial[0] < best[0]-1e-9:
                best = trial
        return best

    cost, steps = dp(0, 0)
    if not math.isfinite(cost):
        raise AlignmentPlanningError('Bounded syllable-to-note alignment failed')
    ni = 0
    mapped = []
    for syl, step in zip(syls,steps):
        t,p,d=notes[ni]
        mapped.append({'word':syl.word,'word_index':syl.word_index,
                       'syllable_index':syl.index,'note_index':ni,'note_count':step,
                       'note_start_s':round(float(t)*60/bpm,4),
                       'note_duration_s':round(sum(times[ni:ni+step]),4),
                       'stressed':syl.stressed,'melisma':step>1})
        ni += step
    return mapped, {'cost':round(cost,4),'syllables':n,'notes':count,'status':'aligned_symbolically'}


def _split_vocal_notes(abc: str, needed: int, *, limits_per_section: list[int] | None = None, limit: int = 64) -> tuple[str,int]:
    """Make repeated-pitch attacks, preserving bar beat grids and Ins exactly.

    Each split changes articulation (not an exact melody event invariant), is
    explicit in the report, and has bounded modification cost.
    """
    if needed <= 0: return abc,0
    lines = abc.splitlines()
    score = parse_abc(abc)
    candidates = []
    regions = score_sections(abc)
    bar_cursor = 0
    for idx,voice in sorted(score.music_lines.items()):
        if voice != 'Vocal': continue
        section_index = next((i for i,sec in enumerate(regions)
                              if sec.first_bar <= bar_cursor < sec.last_bar),None)
        original = lines[idx]
        bar_cursor += _expanded_bar_count(original)
        if section_index is None: continue
        tokens = list(ABC_TOKEN.finditer(original))
        for m in tokens:
            if m.group('note') in (None,'z') or m.group('tie') == '-': continue
            if m.group('chord') is not None or m.group('key') is not None: continue
            units = int(m.group('duration') or '1')
            if units < 2: continue
            prior = original[:m.start()]
            if prior.endswith('-'): continue
            for first in range(units//2,0,-1):
                second = units-first
                if first in DURATIONS and second in DURATIONS:
                    candidates.append((units,idx,m.start(),m.end(),m,first,second,section_index)); break
    candidates.sort(key=lambda x:(-x[0],x[1],x[2]))
    if limits_per_section is None:
        selected = candidates[:min(needed,limit)]
    else:
        quotas = list(limits_per_section)
        selected=[]
        for candidate in candidates:
            section_idx = candidate[-1]
            if quotas[section_idx] > 0 and len(selected)<limit:
                selected.append(candidate)
                quotas[section_idx] -= 1
    per_line = {}
    for units,idx,start,end,m,a,b,_section in selected:
        head = (m.group('acc') or '')+(m.group('note') or '')+(m.group('oct') or '')
        replacement = head + str(a) + head + str(b)
        per_line.setdefault(idx,[]).append((start,end,replacement))
    for idx, edits in per_line.items():
        for start,end,replacement in sorted(edits,reverse=True):
            lines[idx] = lines[idx][:start]+replacement+lines[idx][end:]
    produced = '\n'.join(lines)+('\n' if abc.endswith('\n') else '')
    after = parse_abc(produced)
    before = score.voices
    if (after.bpm != score.bpm or after.voices['Ins'].notes != before['Ins'].notes or
        after.voices['Ins'].bars != before['Ins'].bars or
        after.voices['Vocal'].bars != before['Vocal'].bars or
        after.voices['Vocal'].time != before['Vocal'].time):
        raise AlignmentPlanningError('Vocal note split violated native staff/tempo preservation')
    return produced,len(selected)




def extend_score_by_native_groups(abc: str, lyrics: str, *, max_duration: float = 0.0,
                                  max_added_groups: int = 96,
                                  max_growth_ratio: float = 4.0) -> tuple[str,int]:
    """Extend only by *complete* safe Vocal+Ins groups at a section boundary.

    Keeps notes within each copied group intact, including accompaniment.
    Does not alter original lyric words. All edits are marked as arrangement
    changes in the returned report; it does not claim original melody identity.
    """
    parsed = parse_abc(abc)
    sections = score_sections(abc)
    if not sections: return abc, 0
    _planned, lines_for_sections = _sections_lyrics(abc,lyrics,sections)
    deficits = [max(0,len(syllabify(words))-len(sec.notes)) for sec,words in zip(sections,lines_for_sections)]
    if not any(deficits): return abc, 0
    lines = abc.splitlines()
    keys = sorted(parsed.music_lines)
    if len(keys)%2 or any(parsed.music_lines[keys[i]] != 'Vocal' or
                          parsed.music_lines[keys[i+1]] != 'Ins' for i in range(0,len(keys),2)):
        return abc, 0
    groups=[]
    bar_cursor=0
    for i in range(0,len(keys),2):
        vi,ii=keys[i:i+2]
        sec_idx=next((j for j,sec in enumerate(sections)
                      if sec.first_bar<=bar_cursor<sec.last_bar),None)
        count=_expanded_bar_count(lines[vi])
        bar_cursor+=count
        if sec_idx is None or not lines[vi].strip() or '-' in lines[vi] or '-' in lines[ii]:
            continue
        # Never duplicate groups with in-group local key/meter changes.
        if '[K:' in lines[vi] or '[K:' in lines[ii]: continue
        if any(line.startswith(('K:','M:')) for line in lines[vi+1:ii]): continue
        start_beat=parsed.voices['Vocal'].bars[bar_cursor-count][0]
        end_beat=parsed.voices['Vocal'].bars[bar_cursor-1][0]+parsed.voices['Vocal'].bars[bar_cursor-1][1]
        group_notes=sum(1 for n in parsed.voices['Vocal'].notes if start_beat <= n[0] < end_beat)
        if group_notes <= 0: continue
        group_seconds=float((end_beat-start_beat)*60/parsed.bpm)
        groups.append((vi,ii,sec_idx,group_notes,count,group_seconds))
    changes=[]
    remaining=max_added_groups
    nominal=float(parsed.voices['Vocal'].time*60/parsed.bpm)
    for sec_idx,required in enumerate(deficits):
        eligible=[g for g in groups if g[2]==sec_idx]
        if not eligible or not required or not remaining: continue
        # Repeat complete musical motifs at section boundaries; the native score
        # validator enforces the tempo/grid. Budget and required syllables cap
        # the actual repetitions (no arbitrary three-group global ceiling).
        vi,ii,_,onsets,bars,added_seconds=eligible[-1]
        times=min(remaining,math.ceil(required/onsets))
        if max_duration>0:
            available=max(0.0,max_duration-3.0-nominal)
            times=min(times,int(available//max(.01,added_seconds)))
        else:
            # Without explicit budget, allow bounded arrangement growth;
            # candidate scoring penalizes every repeated group.
            original_seconds=float(parsed.voices['Vocal'].time*60/parsed.bpm)
            already_added=nominal-original_seconds
            times=min(times,max(0,int((original_seconds*max_growth_ratio-already_added)//max(.01,added_seconds))))
        if times<=0:continue
        changes.append((ii,lines[vi],lines[ii],times))
        nominal+=times*added_seconds
        remaining-=times
    if not changes: return abc,0
    for ii,vocal,ins,times in sorted(changes,reverse=True):
        snippet=['V: Vocal',vocal,'V: Ins',ins]*times
        lines[ii+1:ii+1]=snippet
    output='\n'.join(lines)+('\n' if abc.endswith('\n') else '')
    candidate=parse_abc(output)
    actual_seconds=float(candidate.voices['Vocal'].time*60/candidate.bpm)
    if (candidate.bpm!=parsed.bpm or candidate.voices['Vocal'].time!=candidate.voices['Ins'].time or
        len(candidate.voices['Vocal'].bars)!=len(candidate.voices['Ins'].bars) or
        (max_duration>0 and actual_seconds>max_duration-3+1e-6)):
        raise AlignmentPlanningError('Repeated native group broke score grid invariant')
    return output,sum(item[3] for item in changes)

def short_tagged_score_excerpt(abc: str, lyrics: str, *, maximum_seconds: float) -> tuple[str, dict] | None:
    """Extract a real labeled motif for a *single, very short, tagged* lyric block.

    A two-word [Chorus] cannot be mapped onto the opening 30 s of an
    unrelated Verse in a five-minute SheetSage score. Prefer the requested
    musical section and rest surplus Vocal *bars*, retaining the accompaniment
    and the complete chosen bar grid. This is an explicitly reported excerpt,
    not a claim to preserve the entire original arrangement.
    """
    from .alignment import lyric_parts, _canonical
    raw_words, parts = _tokens(lyrics)
    sung = [p for p in parts if p.sung_lines]
    if len(sung) != 1 or not sung[0].tag or not 1 <= len(raw_words) <= 12:
        return None
    requested = _canonical(sung[0].tag)
    if requested not in ('verse', 'chorus', 'pre chorus', 'bridge', 'refrain', 'rap'):
        return None
    source = parse_abc(abc)
    lines = abc.splitlines()
    pairs = sorted(source.music_lines)
    if len(pairs) % 2:
        return None
    label = ''
    preceding = 7
    viable = []
    total_syllables = max(1, len(syllabify(lyrics)))
    for i in range(0, len(pairs), 2):
        vi, ii = pairs[i:i+2]
        if source.music_lines[vi] != 'Vocal' or source.music_lines[ii] != 'Ins':
            return None
        headings = [line[2:].strip() for line in lines[preceding+1:vi] if line.startswith('% ')]
        if headings:
            label = headings[-1]
        preceding = ii
        if _canonical(label) != requested:
            continue
        # Native pairs in this dialect consist of a separate V: header and
        # one music line. Skip unusual local metadata rather than creating an
        # invalid score or silently losing time/key changes.
        if vi < 1 or ii != vi + 2 or lines[vi-1].strip() != 'V: Vocal' or lines[ii-1].strip() != 'V: Ins':
            continue
        original_music = lines[vi].strip()
        if '[K:' in original_music or '[M:' in original_music:
            continue
        bars = [bar.strip() for bar in original_music.rstrip('|').split('|') if bar.strip()]
        if not bars:
            continue
        # Keep a genuine first-bar vocal phrase; silence later sung bars so
        # missing lyrics cannot be represented as a full multibar vocal solo.
        # The instrumental part is left byte-identical.
        for retained in range(1, len(bars)+1):
            remaining = sum(_expanded_bar_count(bar+'|') for bar in bars[retained:])
            if remaining < 0:
                continue
            rewritten = '|'.join(bars[:retained])+'|'
            if remaining:
                # Native Z can encode 1..4 measures per symbol.
                while remaining:
                    chunk = min(4,remaining)
                    rewritten += ('Z' if chunk == 1 else 'Z'+str(chunk))+'|'
                    remaining -= chunk
            body = ['% '+label, 'V: Vocal', rewritten, 'V: Ins', lines[ii]]
            candidate = '\n'.join(lines[:8]+body)+'\n'
            try:
                parsed = parse_abc(candidate)
            except ValueError:
                continue
            seconds = float(parsed.voices['Vocal'].time * 60 / parsed.bpm)
            if seconds > maximum_seconds - 3 or seconds < 3:
                continue
            onsets = len(parsed.voices['Vocal'].notes)
            if onsets < 1 or onsets / total_syllables > 3.5:
                continue
            if parsed.voices['Ins'].notes != parse_abc('\n'.join(lines[:8]+['% '+label,'V: Vocal',original_music,'V: Ins',lines[ii]])+'\n').voices['Ins'].notes:
                continue
            # Most note attacks with reasonable syllable fit, then original
            # section order; never select a generic Verse for [Chorus].
            viable.append((abs(onsets/total_syllables - 1.4),i,seconds,candidate,onsets))
    if not viable:
        return None
    viable.sort(key=lambda entry:(entry[0],entry[1]))
    _,group_position,seconds,chosen,onsets = viable[0]
    original_seconds = float(source.voices['Vocal'].time * 60 / source.bpm)
    return chosen, {
        'action':'tagged_short_lyrics_excerpt',
        'reason':'Single explicit lyric section is too short for the complete source form',
        'matched_section':requested,
        'source_duration_s':round(original_seconds,3),
        'output_duration_s':round(seconds,3),
        'selected_source_group_index':group_position//2,
        'retained_vocal_onsets':onsets,
        'total_estimated_lyric_syllables':total_syllables,
        'source_form_preserved':False,
        'instrumental_notes_in_selected_group_preserved':True,
        'lyrics_word_sequence_preserved':True,
    }


def trim_score_to_duration(abc: str, lyrics: str, max_duration: float,
                           *, reserve_seconds: float = 3.0, max_tempo_boost: float = .08) -> tuple[str, dict]:
    """Fit *whole native groups* into a strict ceiling, never deleting words.

    Section end is preferred when cuts yield similar retained durations.
    Rejects unplayable / over-dense candidates. The output is still a symbolic
    arrangement, not a forced-duration or phonetic model.
    """
    original = parse_abc(abc)
    original_seconds = float(original.voices['Vocal'].time*60/original.bpm)
    if max_duration <= 0 or original_seconds <= max_duration - reserve_seconds:
        return abc, {'action':'not_required','source_duration_s':round(original_seconds,3)}
    usable = max_duration - reserve_seconds
    if usable <= 0:
        raise AlignmentPlanningError('max_duration is too short for the 3-second reserved tail')
    source_lines = abc.splitlines()
    instrumentals = sorted(idx for idx,voice in original.music_lines.items() if voice=='Ins')
    candidates=[]
    for index,cut in enumerate(instrumentals):
        prefix = '\n'.join(source_lines[:cut+1])+'\n'
        try:
            score = parse_abc(prefix)
            raw_s = float(score.voices['Vocal'].time*60/score.bpm)
            required_ratio = raw_s/usable
            if required_ratio > 1 + max_tempo_boost + 1e-9:
                continue
            speed = max(1.0,required_ratio)
            target_bpm = max(original.bpm, math.ceil(original.bpm*speed))
            if target_bpm / original.bpm > 1 + max_tempo_boost + 1e-9:
                continue
            trimmed_lines=prefix.splitlines()
            trimmed_lines[4]=f'Q:1/4={target_bpm}'
            result='\n'.join(trimmed_lines)+'\n'
            trimmed=parse_abc(result)
            duration=float(trimmed.voices['Vocal'].time*60/trimmed.bpm)
            if duration > usable + 1e-6: continue
            # Retain the text exactly, only re-allocate its unmodified words.
            sections=score_sections(result)
            if not sections:continue
            _,groups=_sections_lyrics(result,lyrics,sections)
            max_density=max(len(syllabify(g))/max(.01,sec.seconds) for sec,g in zip(sections,groups))
            if max_density > 7.8: continue
            following = source_lines[cut+1:instrumentals[index+1]] if index+1<len(instrumentals) else []
            boundary = index+1 == len(instrumentals) or any(line.startswith('% ') for line in following)
            candidates.append((duration,boundary,result,target_bpm,index+1,max_density))
        except (ValueError,AlignmentPlanningError):
            continue
    if not candidates:
        raise AlignmentPlanningError('Cannot shorten music within max_duration without exceeding '
                                     '8% BPM boost or 7.8 sung syllables/second; lyrics preserved')
    # Prefer maximum retained musical content, then section boundary.
    best = max(candidates,key=lambda c:(c[4],int(c[1]),c[0]))
    duration,boundary,result,new_bpm,groups,density=best
    return result,{'action':'tempo_only_fit' if len(instrumentals)==groups else 'trim_whole_trailing_groups',
                   'source_duration_s':round(original_seconds,3),
                   'output_duration_s':round(duration,3),
                   'groups_removed':len(instrumentals)-groups,
                   'cut_on_section_boundary':boundary,
                   'bpm_before':original.bpm,'bpm_after':new_bpm,
                   'tempo_boost_pct':round(100*(new_bpm/original.bpm-1),3),
                   'max_section_syllables_per_second':round(density,3),
                   'reserved_tail_seconds':reserve_seconds}

def _score_cost(abc:str, lyrics:str) -> tuple[float,dict]:
    sections = score_sections(abc)
    planned,groups = _sections_lyrics(abc,lyrics,sections)
    score = parse_abc(abc)
    total_cost = 0.0
    density = []
    for section,text in zip(sections,groups):
        syls = syllabify(text)
        ratio = len(syls)/max(1,len(section.notes))
        density_s = len(syls)/max(.01,section.seconds)
        _mapping, diagnostics = _align_syllables(section.notes,syls,score.bpm)
        total_cost += diagnostics['cost']+max(0,ratio-1)*4+max(0,density_s-7.8)*15
        density.append({'label':section.label,'syllables':len(syls),'notes':len(section.notes),
                        'seconds':round(section.seconds,3),'syllables_per_second':round(density_s,3),
                        'alignment_cost':diagnostics['cost']})
    return round(total_cost,4), {'sections':density,'total_syllables':sum(x['syllables'] for x in density),
                                  'total_notes':sum(x['notes'] for x in density)}


def optimize(abc: str, lyrics: str, *, mode: str = 'Adaptive',
             max_duration: float = 0.0, strict: bool = True) -> tuple[str,str,dict]:
    """Return symbolic ABC, word-preserving lyrics and an auditable plan.

    If the ABC is invalid, the caller decides about the *official* YuE2 ABC
    regeneration. This component never invents words or claims forced vocals.
    """
    if mode not in MODES:
        raise ValueError('Invalid vocal melody mode: '+str(mode))
    from .instrumental_mode import instrumental_requested, instrumentalize_score, END
    if instrumental_requested(lyrics):
        score, evidence = instrumentalize_score(abc, max_duration=max_duration)
        return score, END, {'engine': 'YuE2 Vocal-Melody Alignment Engine',
                            'version': '1.1', 'mode': 'Instrumental',
                            'instrumental': evidence, 'word_order_unchanged': True,
                            'planned_word_count': 0, 'max_duration_s': max_duration}
    before = parse_abc(abc)
    raw_words,_ = _tokens(lyrics)
    if not raw_words:
        raise AlignmentPlanningError('Target lyrics contain no sung words')
    duration_fit = None
    source_abc = abc
    original_seconds = float(before.voices['Vocal'].time*60/before.bpm)
    # max_duration is a hard ceiling, NOT a request to fill the entire time.
    # Replacement lyrics may be shorter than the source. That alone cannot
    # authorize shortening the supplied arrangement; evaluate syllable density
    # as diagnostic evidence and preserve the musical score where possible.
    syllable_count = len(syllabify(lyrics))
    soft_budget = max(32.0, syllable_count / 2.15 + 30.0)
    sparse_source = original_seconds > soft_budget * 1.65
    effective_budget = float(max_duration) if max_duration > 0 else 0.0
    # The source score is the user's song. Do NOT make a 300-second cover
    # into a 60-second excerpt merely because the replacement text is short.
    # Only the explicit user maximum may shorten music by default. The
    # syllable-based estimate remains a warning, never a hidden hard ceiling.
    # An explicit user Music-adapts mode can still use its own arrangement rules.
    if effective_budget > 0 and original_seconds > effective_budget:
        if mode == 'Lyrics adapt to Music':
            raise AlignmentPlanningError('Lyrics adapt to Music cannot shorten a score exceeding max_duration')
        # Very short, explicitly tagged Lyrics target the matching original
        # section, not the first 30 seconds regardless of what is there.
        # This may produce a short excerpt instead of a full-song cover.
        excerpt = (short_tagged_score_excerpt(abc, lyrics, maximum_seconds=effective_budget)
                   if sparse_source else None)
        if excerpt is not None:
            abc, duration_fit = excerpt
        else:
            abc,duration_fit = trim_score_to_duration(abc,lyrics,effective_budget)
        duration_fit['soft_budget_lyrics_based_s'] = round(soft_budget, 3) if sparse_source else None
        duration_fit['effective_budget_s'] = round(effective_budget, 3)
    proposals = [(abc,'preserve_original_score' if abc == source_abc else 'trim_to_duration',0,0)]
    if mode != 'Lyrics adapt to Music':
        try:
            s,ly = _sections_lyrics(abc, lyrics,score_sections(abc))
            sections = score_sections(abc)
            deficits = [max(0,len(syllabify(text))-len(sec.notes)) for sec,text in zip(sections,ly)]
            deficit = sum(deficits)
            if deficit:
                modified,count = _split_vocal_notes(abc,deficit,limits_per_section=deficits)
                if count:
                    proposals.append((modified,'split_repeated_pitch_notes',count,0))
            # Whole-group extension is an explicit, bounded arrangement edit.
            extended,copies=extend_score_by_native_groups(abc,lyrics,max_duration=max_duration)
            if copies:
                proposals.append((extended,'extend_complete_musical_groups',0,copies))
        except (ValueError,AlignmentPlanningError):
            pass
    evaluated = []
    for candidate,action,edits,copies in proposals:
        try:
            score = parse_abc(candidate)
            duration = float(score.voices['Vocal'].time*60/score.bpm)
            if max_duration>0 and duration > max_duration+1e-5:
                continue
            cost,diagnostics = _score_cost(candidate,lyrics)
            # Strong music identity penalty: Adaptive leaves source intact
            # unless splitting demonstrably improves syllable/note fit.
            cost += edits*(0.42 if mode=='Adaptive' else 0.15) + copies*(4.0 if mode=='Adaptive' else 1.5)
            evaluated.append((cost,edits,candidate,action,diagnostics,duration,copies))
        except (ValueError, AlignmentPlanningError):
            continue
    if not evaluated:
        raise AlignmentPlanningError('No native ABC plan fits the maximum duration and target Lyrics')
    evaluated.sort(key=lambda x:(x[0],x[1],x[3]))
    cost,edits,chosen,action,diagnostics,duration,copies = evaluated[0]
    planned,group_texts = _sections_lyrics(chosen,lyrics,score_sections(chosen))
    chosen_words,_ = _tokens(planned)
    if raw_words != chosen_words:
        raise AlignmentPlanningError('Target lyrics word sequence changed unexpectedly')
    scores = score_sections(chosen)
    native_score=parse_abc(chosen)
    mappings=[]
    densest=0.0
    for sec,text in zip(scores,group_texts):
        mapped,info = _align_syllables(sec.notes,syllabify(text),native_score.bpm)
        densest=max(densest,len(mapped)/max(.01,sec.seconds))
        mappings.append({'label':sec.label,'diagnostics':info,'syllable_slots':mapped})
    if densest>7.8 and strict:
        raise AlignmentPlanningError('Syllable density exceeds 7.8 syllables/s in one section; '
                                     'the engine will not invent/repeat/cut words')
    source_duration=float(before.voices['Vocal'].time*60/before.bpm)
    report={'engine':'YuE2 Vocal-Melody Alignment Engine',
            'version':'1.0','mode':mode,'action':action,'score_changed':chosen!=source_abc,
            'repeated_pitch_attacks_added':edits,'full_groups_added':copies,
            'source_duration_s':round(source_duration,3),
            'soft_duration_target_s':round(soft_budget,3) if sparse_source else None,
            'output_duration_s':round(duration,3),'max_duration_s':float(max_duration),
            'input_word_count':len(raw_words),'planned_word_count':len(chosen_words),
            'word_order_unchanged':raw_words==chosen_words,
            'optimization_cost':round(cost,4), 'fitness':diagnostics, 'duration_fit':duration_fit,
            'syllable_mapping':mappings,
            'lexical_contamination_guard':'Target and reference lyrics are isolated by distinct graph inputs; AR generation remains probabilistic',
            'limits':'symbolic score/lyrics planning only, not enforced phoneme-to-note acoustic alignment',
            'warnings':(['Singing ASR/listening required to verify actual generated words'] +
                        (['Syllable density over 7.8/s: lower confidence'] if densest>7.8 else []))}
    return chosen,planned,report


class YuE2VocalMelodyAlignmentEngine:
    """Optional standalone inspection node; production routes call optimize directly."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'abc': ('STRING', {'forceInput': True}),
            'lyrics': ('STRING', {'forceInput': True}),
            'mode': (list(MODES), {'default':'Adaptive'}),
            'max_duration': ('FLOAT', {'default':0.0,'min':0.0,'max':900.0}),
            'strict': ('BOOLEAN', {'default':True}),
        }}

    RETURN_TYPES = ('STRING','STRING','STRING')
    RETURN_NAMES = ('abc','lyrics','alignment_report')
    FUNCTION = 'plan'
    CATEGORY = 'audio/YuE2'
    DESCRIPTION = 'Symbolically align immutable Lyrics with native YuE2 Vocal/Ins ABC; results are not forced acoustic synthesis.'

    def plan(self, abc, lyrics, mode='Adaptive', max_duration=0.0, strict=True):
        # The optional standalone planner now uses the exact same hard gates
        # as the Standard/Reference Voice production route. Keep the old
        # optimize() function available for diagnostic score experiments,
        # but do not expose its heuristic rejection as a normal workflow gate.
        from .cover_contract import prepare_cover
        from .lyric_contract import with_terminal_end
        plan = prepare_cover(abc, lyrics, max_duration=max_duration,
                             origin='standalone_native_abc')
        evidence = dict(plan.report)
        evidence['engine'] = 'YuE2 Vocal-Melody Alignment Engine'
        evidence['mode'] = mode
        evidence['strict'] = bool(strict)
        return plan.abc, with_terminal_end(plan.lyrics), json.dumps(evidence, ensure_ascii=False, default=str)
