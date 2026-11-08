"""Single production contract for YuE2 source-score covers.

A source ABC is musical evidence, not a syllable-count target.  The only
hard gates are native ABC validity, a REAL user duration limit, preserved
lyrics words, and coherent score/lyric sections.  The model itself determines
actual phonetic timing; score/planning metrics cannot guarantee sung words.

No SVC, no acoustic model replacement and no implicit YuE2GenerateABC for
valid SheetSage2 scores.  Both Standard and Reference Voice use this contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .alignment import _canonical, lyric_parts
from .lyric_contract import _tokens, measure
from .section_contract import native_sections, materialize_native_blocks
from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc


class CoverContractError(ValueError):
    """An actual native score/lyrics invariant was violated."""


@dataclass(frozen=True)
class CoverPlan:
    abc: str
    lyrics: str
    report: dict


def _duration(abc: str) -> float:
    score = parse_abc(abc)
    return float(score.voices['Vocal'].time * 60 / score.bpm)


def _candidate_starts(abc: str):
    """Native pair endings, without reconstructing notes or breaking ties."""
    score = parse_abc(abc)
    ordered = sorted(score.music_lines)
    if len(ordered) % 2 or any(score.music_lines[ordered[i]] != 'Vocal' or
                                  score.music_lines[ordered[i + 1]] != 'Ins'
                                  for i in range(0, len(ordered), 2)):
        raise CoverContractError('ABC is not made of paired Vocal/Ins groups')
    lines = abc.splitlines()
    for i in range(0, len(ordered), 2):
        yield '\n'.join(lines[:ordered[i+1] + 1]) + '\n', i // 2 + 1


def _fit_whole_score_by_tempo(abc: str, original_s: float, max_duration: float,
                              *, max_bpm_increase: float = 0.08):
    """Fit a small duration overflow by changing the *header BPM only*.

    Native YuE2 accepts integer Q:1/4 BPM. Never touch note/rest events,
    meter, keys, section order or lyrics. Fail cleanly if >8% is needed.
    The parser is the final authority: check the entire event inventory.
    """
    if max_duration <= 0 or not math.isfinite(max_duration):
        return None
    parsed = parse_abc(abc)
    original_bpm = parsed.bpm
    needed = max(original_bpm + 1, math.ceil(original_bpm * original_s / max_duration - 1e-9))
    if needed / original_bpm > 1.0 + max_bpm_increase + 1e-12:
        return None
    lines = abc.splitlines(keepends=True)
    if len(lines) < 8 or not re.fullmatch(r'Q:1/4=\d+\r?\n?', lines[4]):
        return None
    newline = '\r\n' if lines[4].endswith('\r\n') else '\n' if lines[4].endswith('\n') else ''
    lines[4] = f'Q:1/4={needed}' + newline
    candidate = ''.join(lines)
    reparsed = parse_abc(candidate)
    for voice in ('Vocal', 'Ins'):
        before, after = parsed.voices[voice], reparsed.voices[voice]
        if (before.notes != after.notes or before.chords != after.chords or
            before.bars != after.bars or before.keys != after.keys or
            before.time != after.time):
            raise CoverContractError('Tempo-only fit changed native score events')
    fitted_s = float(reparsed.voices['Vocal'].time * 60 / reparsed.bpm)
    if fitted_s > max_duration + 1e-6:
        return None
    return candidate, {
        'action': 'tempo_fit_full_score', 'original_seconds': round(original_s, 3),
        'selected_seconds': round(fitted_s, 3), 'excerpt': False,
        'original_bpm': original_bpm, 'selected_bpm': needed,
        'bpm_increase_percent': round(100.0 * (needed / original_bpm - 1.0), 4),
        'note_events_modified': False, 'section_order_preserved': True,
        'warning': 'Full score retained; playback tempo increased by up to 8% to fit the maximum.'}


def _fit_only_real_duration(abc: str, lyrics: str, max_duration: float) -> tuple[str, dict]:
    """Keep a playable full score or a native group excerpt under the explicit cap.

    No word-count, syllable density, arbitrary fixed tail, or soft-duration gate.
    Do not manufacture tempo or split a tied group.
    """
    original_s = _duration(abc)
    if max_duration <= 0 or original_s <= max_duration + 1e-6:
        return abc, {'action': 'preserve_full_score', 'original_seconds': round(original_s, 3),
                     'selected_seconds': round(original_s, 3), 'excerpt': False}
    # A few seconds of overflow should not force loss of the entire song:
    # first attempt a small, auditable BPM adjustment, capped at +8%.
    tempo_fit = _fit_whole_score_by_tempo(abc, original_s, max_duration)
    if tempo_fit is not None:
        return tempo_fit
    # A specifically requested Chorus/Verse is better served by a real
    # section-matched excerpt than by automatically selecting the intro.
    try:
        from .bounded_recovery import recover_score_for_lyrics
        recovered = recover_score_for_lyrics(abc, lyrics, max_duration)
        if recovered:
            candidate, trace = recovered
            if _duration(candidate) <= max_duration + 1e-6:
                return candidate, {'action': 'native_section_excerpt',
                                   'original_seconds': round(original_s, 3),
                                   'selected_seconds': round(_duration(candidate), 3),
                                   'excerpt': True, 'source_trace': trace}
    except (ValueError, IndexError, KeyError):
        pass
    # No arbitrary intro/prefix when the requested vocal section is elsewhere.
    # Such a fallback would silently turn a requested Chorus into an Intro.
    raise CoverContractError(
        f'No contiguous native Vocal/Ins excerpt with matching vocal sections '
        f'fits max_duration={max_duration:g}s (source {original_s:.1f}s). '
        'The source score cannot satisfy this explicit constraint without '
        'altering the music; no invented notes or replaced lyrics were used.')


def _allocate_rich_sections(sections, vocal_indices, clean_parts):
    """Weighted monotone lyric-line layout for a full song.

    Keeps every original line intact, distributes long target texts across
    actual vocal note capacity, and treats user section names as soft
    chronological anchors. Never aggregates all overflow at the last verse.
    This is not a phoneme aligner and cannot guarantee enunciation.
    """
    from .alignment import _syllable_estimate
    from bisect import bisect_left
    units = [(line, part.tag) for part in clean_parts for line in part.sung_lines]
    n, count = len(units), len(vocal_indices)
    if not count or n < count:
        raise CoverContractError('Not enough lyric lines for a line-preserving full-song layout')
    weights = [max(1, _syllable_estimate([line])) for line, _ in units]
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    capacities = [max(1, sections[index].vocal_onsets) for index in vocal_indices]
    total_caps = sum(capacities)
    boundaries = [0]
    prior = 0
    cum_caps = 0
    for j, capacity in enumerate(capacities[:-1]):
        cum_caps += capacity
        fraction = cum_caps / total_caps
        minimum = prior + 1
        maximum = n - (count - j - 1)
        estimate = bisect_left(prefix, prefix[-1] * fraction)
        # Search the full admissible boundary range: the small lexical anchor
        # may make a nearby named verse more appropriate than nearest quota.
        next_label = _canonical(sections[vocal_indices[j+1]].label)
        def cost(boundary):
            next_tag = units[boundary][1]
            anchor = (0.025 if next_tag and
                      (boundary == 0 or units[boundary-1][1] != next_tag) and
                      _canonical(next_tag) == next_label else 0.0)
            return (abs(prefix[boundary]/prefix[-1] - fraction) - anchor,
                    abs(prefix[boundary]/prefix[-1] - fraction), abs(boundary-estimate))
        best = min(range(minimum, maximum+1), key=cost)
        boundaries.append(best)
        prior = best
    boundaries.append(n)
    assigned = [[] for _ in sections]
    decisions = []
    for j, idx in enumerate(vocal_indices):
        start, end = boundaries[j:j+2]
        assigned[idx] = [line for line, _ in units[start:end]]
        tags = list(dict.fromkeys(tag for _, tag in units[start:end] if tag))
        decisions.append({'score_section':idx, 'score_label':sections[idx].label,
                          'method':'weighted_lyric_line_layout',
                          'original_lyric_tags':tags, 'lines':end-start,
                          'estimated_syllables':prefix[end]-prefix[start],
                          'vocal_note_onsets':capacities[j]})
    return assigned, decisions


def _allocate_text(abc: str, lyrics: str) -> tuple[str, dict]:
    """Preserve EVERY target word, with an ABC section for every native section.

    Explicit tags are anchoring preferences, not a license to reject a
    perfectly valid 300-second musical score. Unfilled sections are explicit
    empty sections (with a logged hallucination risk), never reference words.
    """
    sections = native_sections(abc)
    if not sections:
        raise CoverContractError('No ABC sections to carry the replacement Lyrics')
    expected, parts = _tokens(lyrics)
    if not expected:
        raise CoverContractError('Lyrics contain no sung words')
    vocal_indices = [i for i, s in enumerate(sections) if s.vocal_onsets]
    if not vocal_indices:
        raise CoverContractError('ABC contains no vocal notes for the target Lyrics')
    clean_parts = [p for p in parts if p.sung_lines]
    if not clean_parts:
        raise CoverContractError('Lyrics contain no sung phrases')
    assigned: list[list[str]] = [[] for _ in sections]
    decisions = []
    warnings = []
    # A native one-to-one Lyrics document (possibly with many blank vocal
    # blocks) is idempotent: it must not be redistributed on the next pass.
    tagged = [p for p in parts if p.tag and _canonical(p.tag) != 'end']
    if len(tagged) == len(sections) and all(_canonical(p.tag) == _canonical(s.label)
                                              for p, s in zip(tagged, sections)):
        for i, p in enumerate(tagged):
            if p.sung_lines and i not in vocal_indices:
                raise CoverContractError('Words in a purely instrumental native section')
            assigned[i] = p.sung_lines[:]
        decisions.append({'method': 'idempotent_native_blocks'})
    else:
        has_named = any(p.tag for p in clean_parts)
        rich_lines = sum(len(p.sung_lines) for p in clean_parts)
        # A full-song text with many lines must not be squeezed into the last
        # score section merely because the number/order of its tags differs
        # from SheetSage2's section boundaries. Preserve punctuation and all
        # lines while spreading the text over the existing vocal note grid.
        if (len(clean_parts) >= 2 and len(vocal_indices) >= 2 and
            rich_lines >= len(vocal_indices) and
            len(expected) >= max(20, 4*len(vocal_indices))):
            assigned, decisions = _allocate_rich_sections(sections, vocal_indices, clean_parts)
            warnings.append('Full-song Lyrics were distributed over available vocal passages '
                            'using note capacities and section-label hints; this is an '
                            'approximation, not forced phoneme alignment.')
        elif has_named:
            pointer = 0
            for part in clean_parts:
                label = _canonical(part.tag) if part.tag else None
                eligible = [idx for idx in vocal_indices if idx >= pointer]
                matching = [idx for idx in eligible
                            if label and _canonical(sections[idx].label) == label]
                if matching:
                    selected = matching[0]
                    method = 'matching_named_section'
                elif eligible:
                    selected = eligible[0]
                    method = 'nearest_available_vocal_section'
                    if label:
                        warnings.append(f'No subsequent [{part.tag}] score section: '
                                        f'placed words in %{sections[selected].label}; '
                                        'original Lyrics tag preserved as mapping evidence.')
                else:
                    selected = vocal_indices[-1]
                    method = 'append_to_last_vocal_section'
                    warnings.append('More sung Lyrics blocks than available vocal sections; '
                                    'the remaining phrases share the final section.')
                assigned[selected].extend(part.sung_lines)
                decisions.append({'lyric_tag': part.tag, 'score_section': selected,
                                  'score_label': sections[selected].label, 'method': method})
                pointer = selected + 1
        else:
            # A line-for-line allocation when possible, preserving boundaries
            # and order. For very short text, allocate words rather than
            # requiring at least one entire line for each vocal section.
            from .lyric_contract import _split_weighted
            lines = [line for part in clean_parts for line in part.sung_lines]
            if len(lines) >= len(vocal_indices):
                units = lines
                method = 'whole_lines'
            else:
                units = expected
                method = 'whole_words'
            caps = [max(1, sections[i].vocal_onsets) for i in vocal_indices]
            allocated = _split_weighted(units, caps)
            for i, phrases in zip(vocal_indices, allocated):
                assigned[i] = phrases if method == 'whole_lines' else ([' '.join(phrases)] if phrases else [])
            decisions.append({'method': method, 'vocal_sections': len(vocal_indices)})
    blocks=[]
    for i,section in enumerate(sections):
        label=section.label.strip() or 'Verse'
        if any(c in label for c in '\n\r[]'):
            raise CoverContractError(f'Invalid score section heading: {label!r}')
        blocks.append('['+label+']\n'+'\n'.join(assigned[i]))
    allocated='\n\n'.join(blocks).rstrip()+'\n'
    produced, proof = materialize_native_blocks(abc, allocated)
    actual, _ = _tokens(produced)
    if actual != expected:
        raise CoverContractError('Target word sequence changed during section allocation')
    blank=sum(bool(sections[i].vocal_onsets) and not assigned[i] for i in range(len(sections)))
    if blank:
        warnings.append(f'{blank} vocal sections have no target Lyrics; YuE2 may improvise '
                        'words in these passages. Acoustic output needs listening/ASR.')
    return produced, {'method': 'lyrics_first_native_sections',
                      'native_sections': len(sections), 'vocal_sections': len(vocal_indices),
                      'empty_vocal_sections': blank, 'empty_vocal_section_indices':
                      [i for i,s in enumerate(sections) if s.vocal_onsets and not assigned[i]],
                      'target_word_count': len(expected),
                      'words_identical_and_in_order': True, 'mapping': decisions,
                      'warnings': warnings, 'native_proof': proof}


def prepare_cover(abc: str, lyrics: str, *, max_duration: float = 0.0,
                  origin: str = 'source_sheet_score') -> CoverPlan:
    """Production scorer for both Standard and Reference Voice, no CLIP inference."""
    if not math.isfinite(float(max_duration)) or max_duration < 0:
        raise CoverContractError('Invalid max_duration')
    original = parse_abc(abc)
    if not original.voices['Vocal'].notes:
        raise CoverContractError('No vocal note onsets in the supplied source ABC')
    if original.voices['Vocal'].time != original.voices['Ins'].time:
        raise CoverContractError('Source ABC Vocal/Ins duration mismatch')
    selected, duration = _fit_only_real_duration(abc, lyrics, float(max_duration))
    output, lyric_report = _allocate_text(selected, lyrics)
    from .empty_vocal_arrangement import revoice_empty_sections
    revoiced, instrumental_plan = revoice_empty_sections(
        selected, lyric_report['empty_vocal_section_indices'])
    if revoiced != selected:
        # Revalidate lyrics against the final score: the previously empty
        # vocal groups are now genuinely instrumental, not sung without text.
        revalidated, proof = materialize_native_blocks(revoiced, output)
        if _tokens(revalidated)[0] != _tokens(lyrics)[0]:
            raise CoverContractError('Instrumental continuation changed target word order')
        output = revalidated
        selected = revoiced
        lyric_report['native_proof_after_instrumental_arrangement'] = proof
        lyric_report['empty_vocal_sections_after_arrangement'] = proof['unfilled_vocal_sections']
        lyric_report['warnings'] = [warn for warn in lyric_report['warnings']
            if not warn.startswith(str(lyric_report['empty_vocal_sections'])+' vocal sections')]
        lyric_report['warnings'].append(
            f"{instrumental_plan['groups_revoiced']} unfilled vocal groups moved to Ins "
            "without changing musical note events; sung words remain unchanged.")
        if proof['unfilled_vocal_sections']:
            lyric_report['warnings'].append(
                f"{proof['unfilled_vocal_sections']} vocal sections still lack target "
                "Lyrics and could generate unrequested words.")
    else:
        lyric_report['empty_vocal_sections_after_arrangement'] = lyric_report['empty_vocal_sections']
    # A source score is allowed to contain melismas, instrumental breaks and
    # more/fewer melody notes than words. These ratios are NOT model controls.
    advisory = measure(selected, lyrics, max_duration=max_duration if max_duration > 0 else None)
    from .diction_diagnostics import diction_pressure
    clarity = diction_pressure(selected, output)
    if clarity['status'] == 'estimated_symbolic_only' and clarity['risk_section_count']:
        lyric_report['warnings'].append(
            f"{clarity['risk_section_count']} target sections have dense word/syllable allocation "
            'relative to their score time/note onsets. This may reduce sung intelligibility; '
            'the threshold is advisory, not a generation gate.')
    report = {'score_source': origin, 'decision': 'native_lyrics_first_cover',
              'score_durations': duration, 'max_duration_seconds': float(max_duration),
              'section_contract': lyric_report, 'diction_advisory': clarity,
              'fitness_advisory': advisory.__dict__,
              'instrumental_arrangement': instrumental_plan,
              'voice_engine': 'native_yue2_prefix_unchanged',
              'auto_abc_generation_required': False,
              'lexical_fidelity_certified': False,
              'warnings': lyric_report['warnings'] +
                          (['A valid symbolic score cannot guarantee acoustically sung Lyrics.'])}
    return CoverPlan(selected, output, report)
