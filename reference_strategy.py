"""Controlled Reference Voice experiment parameters for native YuE2.

The semantic prefix is NOT a pure speaker embedding: it contains linguistic
information. These controls reduce an overly dominant prefix, keep reference
ASR/ABC/audio synchronized, and prevent an unbounded AR continuation. They do
not claim to make a stochastic music model lexically deterministic.
"""
from __future__ import annotations

import math

LANGUAGES = ('auto', 'Italian', 'English', 'Spanish', 'French', 'German',
             'Portuguese', 'Chinese', 'Japanese', 'Korean')
CLIP_POLICIES = ('auto', 'full', 'limit')
CONDITIONING_MODES = ('melody', 'full')


def effective_style(user_style: str, target_lyrics: str, target_language: str = 'auto') -> tuple[str, dict]:
    """Preserve supplied style and append a compact target-language cue.

    Lyrics are never copied into Style. Style is guidance, not a word-accuracy
    enforcement mechanism. Report only metadata, not potentially private text.
    """
    from .lyric_contract import _tokens
    if target_language not in LANGUAGES:
        raise ValueError('Unsupported target_language: ' + str(target_language))
    words, _ = _tokens(target_lyrics)
    if not words:
        raise ValueError('Reference Voice target has no sung lyrics')
    raw = str(user_style or '').strip()
    language = (target_language + ' language vocals, ') if target_language != 'auto' else ''
    cue = language + 'clear melodic lead vocal, articulate the supplied target lyrics'
    final = (raw.rstrip(' ,') + ', ' + cue) if raw else cue
    return final, {'user_style_provided':bool(raw), 'style_generated_from_blank':not bool(raw),
                   'target_language':target_language, 'effective_style_nonempty':bool(final),
                   'lyrics_injected_into_style':False, 'style_length':len(final)}


def reference_generation_budget(target_score_seconds: float | None,
                                max_duration: float,
                                *, expansion: float = 0.08, tail_seconds: float = 4.0,
                                policy: str = 'score_bounded') -> tuple[float, dict]:
    """Return an AR ceiling for *target* audio, excluding reference prefix.

    YuE2 may terminate before this ceiling; it is not a target length nor a
    guarantee that it will follow the supplied words.
    """
    if policy not in ('score_bounded','user_maximum'):
        raise ValueError('Unknown reference budget policy: '+str(policy))
    ceiling=float(max_duration)
    if not math.isfinite(ceiling) or not 1 <= ceiling <= 900:
        raise ValueError('max_duration must be finite, between 1 and 900 seconds')
    score = None if target_score_seconds is None else float(target_score_seconds)
    if score is not None and (not math.isfinite(score) or score <= 0):
        raise ValueError('Invalid planned target duration')
    if score is not None and score > ceiling + 0.01:
        raise ValueError(f'Target ABC ({score:.2f}s) exceeds max_duration ({ceiling:.2f}s)')
    budget=ceiling if score is None or policy=='user_maximum' else min(ceiling, float(math.ceil(score*(1+expansion)+tail_seconds)))
    return budget, {'policy':'bounded_by_target_abc' if policy=='score_bounded' else 'legacy_user_maximum', 'target_score_seconds':score,
                    'user_max_duration_s':ceiling, 'target_semantic_ceiling_s':budget,
                    'expansion_allowance':expansion,'tail_allowance_s':tail_seconds,
                    'is_exact_length':False}


def select_reference_clip(audio: dict, reference_lyrics: str = '',
                          clip_policy: str = 'auto', clip_seconds: float = 30.0):
    """Select a *leading* voiced clip, shared by Whisper, SheetSage and tokenizer.

    In auto mode, manually provided reference lyrics keep the full clip because
    cutting audio without cutting its manually supplied transcript is unsafe.
    The boundary is chosen by lower local energy and faded; this is NOT speech
    recognition or a guaranteed musical phrase boundary.
    """
    import torch
    if clip_policy not in CLIP_POLICIES:
        raise ValueError('Unknown reference clip policy: ' + str(clip_policy))
    limit=float(clip_seconds)
    if not math.isfinite(limit) or not 15 <= limit <= 60:
        raise ValueError('reference_clip_seconds must be between 15 and 60')
    if not isinstance(audio,dict) or 'waveform' not in audio or 'sample_rate' not in audio:
        raise ValueError('Expected a ComfyUI AUDIO object for reference clip')
    x=audio['waveform']; sr=int(audio['sample_rate'])
    if getattr(x,'ndim',0)!=3 or x.shape[0]!=1 or x.shape[1] not in (1,2) or sr<8000:
        raise ValueError('Reference clip must be [1, mono|stereo, samples]')
    total=int(x.shape[-1]); total_seconds=total/sr
    if not 15 <= total_seconds <= 60:
        raise ValueError('Reference recording must be between 15 and 60 seconds')
    manual=bool(str(reference_lyrics or '').strip())
    should_clip=(total_seconds > limit + 0.25 and clip_policy == 'limit') or (
        total_seconds > limit + 0.25 and clip_policy == 'auto' and not manual)
    if not should_clip:
        reason=('manual_reference_lyrics_preserve_whole_audio' if manual and clip_policy=='auto'
                else 'full_reference_selected')
        return audio, {'policy':clip_policy,'requested_limit_s':limit,
                       'original_seconds':round(total_seconds,3),
                       'selected_seconds':round(total_seconds,3), 'cropped':False,
                       'boundary_method':'none','reason':reason,
                       'reference_lyrics_must_match_selected_clip':True}
    # Inspect 100ms candidate windows only near the user's requested limit.
    # Preserve almost the entire requested duration. Choosing a silence 4s
    # early caused 26.1s selected clips to be zero-padded to 30s by the
    # native 25Hz/reference-score boundary, polluting the forced prefix.
    earliest=max(15., limit-0.5)
    best=(float('inf'),int(limit*sr))
    for step in range(41):
        second=earliest+(limit-earliest)*step/40
        end=min(total,int(round(second*sr)))
        start=max(0,end-int(round(.12*sr)))
        local=x[...,start:end].detach().float().cpu()
        rms=float(torch.sqrt((local*local).mean()+1e-12)) if local.numel() else float('inf')
        # Very small preference for a longer prefix when RMS is comparable.
        score=rms + (limit-second)*0.0001
        if score < best[0]:
            best=score,end
    cut=max(1,min(total,best[1]))
    kept=x[...,:cut].clone()
    fade=min(cut,int(round(.025*sr)))
    if fade>1:
        kept[...,-fade:]*=torch.linspace(1.,0.,fade,device=kept.device,dtype=kept.dtype)
    selected={'waveform':kept.contiguous(),'sample_rate':sr}
    return selected, {'policy':clip_policy,'requested_limit_s':limit,
                      'original_seconds':round(total_seconds,3),
                      'selected_seconds':round(cut/sr,3),'cropped':True,
                      'boundary_method':'minimum_local_rms_near_requested_limit',
                      'reason':'auto_transcribed_reference' if clip_policy=='auto' else 'explicit_limit',
                      'reference_lyrics_must_match_selected_clip':True,
                      'warning':('With manual reference lyrics, provide only words sung in this selected clip.'
                                 if manual else 'Automatic transcription is performed on this selected clip only.')}


def score_for_conditioning(abc: str, mode: str) -> tuple[str, dict]:
    """In melody mode remove *only* ABC chord annotations, as YuE2 recommends.

    The native ABC composer uses the chord-bearing two-voice representation
    to check score integrity. The conditioning representation is allowed to be
    melody-only without changing pitch, meter, note onsets or section order.
    """
    if mode not in CONDITIONING_MODES:
        raise ValueError('Unknown YuE2 conditioning mode')
    if mode == 'full' or '"' not in abc:
        return abc, {'mode':mode,'chords_removed':0,'vocal_notes_preserved':True,
                     'note_grid_preserved':True}
    from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc
    import re
    parsed=parse_abc(abc)
    if not parsed.voices['Vocal'].chords:
        return abc, {'mode':mode,'chords_removed':0,'vocal_notes_preserved':True,
                     'note_grid_preserved':True}
    lines=abc.splitlines()
    for line_no in parsed.music_lines:
        lines[line_no]=re.sub(r'"[^"\n]*"','',lines[line_no])
    adjusted='\n'.join(lines)+('\n' if abc.endswith('\n') else '')
    after=parse_abc(adjusted)
    for voice in ('Vocal','Ins'):
        before_v=parsed.voices[voice]; after_v=after.voices[voice]
        if (before_v.notes != after_v.notes or before_v.bars != after_v.bars
                or before_v.keys != after_v.keys):
            raise RuntimeError('Melody-only preparation changed musical events')
    if after.voices['Vocal'].chords or after.voices['Ins'].chords:
        raise RuntimeError('Melody-only preparation retained chord annotations')
    return adjusted,{'mode':mode,'chords_removed':len(parsed.voices['Vocal'].chords),
                     'vocal_notes_preserved':True,'note_grid_preserved':True}
