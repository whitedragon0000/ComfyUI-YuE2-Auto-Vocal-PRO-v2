"""Section-accurate YuE2 reference-score composition, same ComfyUI process.

Uses Ornn8/yue2-cover's ABC composer and strict target-score checks. The
reference audio is stretched without pitch change and aligned to the song
score before encoding into the 25 Hz real-audio semantic-token stream.
"""
from __future__ import annotations
import gc
import re
import math
from pathlib import Path
from fractions import Fraction

from .model_store import locations
from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc
from .vendor.yue2_cover_main.yue2_cover.compose_score import compose, sections


def waveform_mono(audio, target_sr=24000):
    import numpy as np
    from scipy.signal import resample_poly
    if not isinstance(audio,dict) or 'waveform' not in audio:
        raise ValueError('Expected a ComfyUI AUDIO object')
    t=audio['waveform'].detach().float().cpu()
    if t.ndim != 3 or t.shape[0] != 1 or t.shape[1] not in (1,2):
        raise ValueError('Reference AUDIO must be [1, 1|2, samples]')
    sr=int(audio['sample_rate'])
    if sr < 8000 or sr > 192000:
        raise ValueError('Invalid audio sample rate')
    mono=t.numpy()[0].mean(axis=0)
    if len(mono)<sr:
        raise ValueError('Reference recording must be at least one second')
    if not np.isfinite(mono).all():
        raise ValueError('Reference contains NaN/Inf')
    if sr!=target_sr:
        d=math.gcd(sr,target_sr)
        mono=resample_poly(mono,target_sr//d,sr//d)
    return mono.astype('float32')


def first_sung_note_seconds(audio, sr):
    """Coarse audible-onset estimate; manual first_note_seconds remains authoritative."""
    import numpy as np
    size=max(240,int(sr*.025))
    pad=(-len(audio))%size
    x=np.pad(audio,(0,pad)).reshape(-1,size)
    rms=np.sqrt(np.mean(x*x,axis=1)+1e-12)
    if np.max(rms)<1e-4:
        raise ValueError('Reference audio appears silent')
    threshold=max(.008,float(np.max(rms))*.15)
    above=rms>=threshold
    for i in range(max(0,len(above)-4)):
        if above[i:i+4].sum()>=3:
            return i*size/sr
    return int(np.argmax(rms))*size/sr


def lyrics_by_abc_section(abc, lyrics):
    """Section-safe Lyrics assembly with word-order invariants.

    The previous algorithm assigned the *first* input lyrics block to the first
    vocal score section and left the remaining vocal sections wordless. It
    caused unconditioned singing and words copied from the reference prompt.
    """
    from .lyric_contract import automatic_lyrics
    planned, diagnostics = automatic_lyrics(abc, lyrics)
    if diagnostics['status'] != 'planned':
        raise ValueError('Reference score/lyrics could not be aligned: ' + diagnostics['reason'])
    from .lyric_contract import with_instrumental_tail
    return with_instrumental_tail(abc, planned)


def compose_reference(target_abc, reference_abc, lyrics, reference_lyrics, reference_audio,
                      first_note_seconds=-1.0):
    """Pure in-memory score composition plus pitch-preserving reference audio align."""
    import numpy as np
    import librosa
    import torch
    target=parse_abc(target_abc)
    reference=parse_abc(reference_abc)
    if not reference.voices['Vocal'].notes:
        raise ValueError('Reference ABC has no vocal melody after routing')
    if not target.voices['Vocal'].notes:
        raise ValueError('Target ABC has no vocal melody after routing')
    rate=float(Fraction(target.bpm,reference.bpm))
    if not 0.25 <= rate <= 4.0:
        raise ValueError('The vocal reference requires more than 4× tempo conversion; provide a different clip')
    mono=waveform_mono(reference_audio,24000)
    source_sec=len(mono)/24000
    if not 15 <= source_sec <= 60:
        raise ValueError('Reference singing clip should be 25–40 seconds (accepted 15–60 s)')
    joined,ref_norm,target_norm,invariant,timing=compose(reference_abc,target_abc,reference_audio_seconds=source_sec)
    if not invariant['pass']:
        raise ValueError('Reference composition changed the target musical notes')
    if abs(rate-1)>1e-5:
        mono=librosa.effects.time_stretch(mono,rate=rate)
    onset_audio=(float(first_note_seconds) if first_note_seconds>=0 else first_sung_note_seconds(waveform_mono(reference_audio),24000))/rate
    reference_first_q=reference.voices['Vocal'].notes[0][0]
    expected_first=float(reference_first_q)*60/target.bpm
    shift=expected_first-onset_audio
    shift_samples=int(round(shift*24000))
    if shift_samples>=0:
        mono=np.pad(mono,(shift_samples,0))
    elif -shift_samples<len(mono):
        mono=mono[-shift_samples:]
    else:
        raise ValueError('Reference first-note alignment exceeds clip length')
    prefix_frames=int(timing['reference_codec_frames'])
    target_samples=prefix_frames*(24000//25)
    # Never silently trim an entire sung phrase: allow only 100 ms rounding slack.
    if len(mono)>target_samples+2400:
        raise ValueError(f'Reference audio ({len(mono)/24000:.2f}s) exceeds the normalized score boundary ({target_samples/24000:.2f}s); adjust reference timing')
    if len(mono)<target_samples:
        mono=np.pad(mono,(0,target_samples-len(mono)))
    else:
        mono=mono[:target_samples]
    aligned={'waveform':torch.from_numpy(mono.astype('float32').copy())[None,None], 'sample_rate':24000}
    ref_section_lyrics=lyrics_by_abc_section(ref_norm,reference_lyrics)
    target_section_lyrics=lyrics_by_abc_section(target_norm,lyrics)
    from .lyric_contract import with_terminal_end
    composite_lyrics=with_terminal_end(ref_section_lyrics.rstrip()+'\n\n'+target_section_lyrics)
    from .lyric_contract import _tokens
    reference_words,_ = _tokens(reference_lyrics)
    target_words,_ = _tokens(lyrics)
    composite_words,_ = _tokens(composite_lyrics)
    if not target_words or not reference_words or composite_words != reference_words + target_words:
        raise ValueError('Reference/target lyric boundary violation: composite word sequence invalid')
    report={'reference_original_bpm':reference.bpm,'target_bpm':target.bpm,
        'tempo_stretch_rate':rate,'reference_first_note_offset_s':onset_audio,
        'score_expected_first_note_s':expected_first,'alignment_shift_s':shift,
        'reference_codec_frames':prefix_frames,'reference_prefix_seconds':prefix_frames/25,
        'composite_target_verified':invariant['pass'],
        'reference_target_lyrics_strictly_concatenated':True,
        'reference_words':len(reference_words),'target_words':len(target_words),
        'target_lyrics_from_reference':False,
        'target_nominal_seconds':timing['target_seconds'],
        'reference_audio_source_seconds':source_sec,
        'tempo_quality_warning': 'Large BPM mismatch may reduce voice articulation' if not 0.8 <= rate <= 1.25 else None,
        'risk':'Automatic first-note detection is approximate; specify first_note_seconds if reference voice sounds misaligned.'}
    return joined,composite_lyrics,aligned,prefix_frames/25,report


class YuE2NativeReferencePlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'target_abc':('STRING',{'forceInput':True}),
            'reference_abc':('STRING',{'forceInput':True}),
            'lyrics':('STRING',{'forceInput':True}),
            'reference_lyrics':('STRING',{'forceInput':True}),
            'reference_audio':('AUDIO',),
            'first_note_seconds':('FLOAT',{'default':-1.0,'min':-1.0,'max':60.0,'step':0.01}),
            'progress_node_id':('STRING',{'default':''}),
            'target_plan_report':('STRING',{'default':''}),
        }}
    RETURN_TYPES=('STRING','STRING','AUDIO','FLOAT','STRING')
    RETURN_NAMES=('abc','lyrics','aligned_reference_audio','prefix_seconds','report')
    FUNCTION='build';CATEGORY='audio/YuE2'
    def build(self,target_abc,reference_abc,lyrics,reference_lyrics,reference_audio,first_note_seconds=-1,progress_node_id='',target_plan_report=''):
        import json
        from .reference_progress import report_progress
        report_progress(progress_node_id,'score_alignment_started')
        outputs=compose_reference(target_abc,reference_abc,lyrics,reference_lyrics,reference_audio,first_note_seconds)
        report_progress(progress_node_id,'score_alignment_completed')
        report=dict(outputs[4])
        if target_plan_report:
            try: report['automatic_target_score']=json.loads(target_plan_report)
            except (ValueError,TypeError): report['automatic_target_score_raw']=str(target_plan_report)
        return (*outputs[:4],json.dumps(report,ensure_ascii=False,indent=2))
