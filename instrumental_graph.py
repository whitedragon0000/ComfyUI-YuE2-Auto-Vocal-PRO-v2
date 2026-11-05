"""Native ComfyUI graph for an instrumental YuE2 cover, no singer prefix.

The reference-voice AR/MERT/NAR pipeline is intentionally not touched. If the
user requests *no words*, a voice prefix would be semantically inappropriate,
so reference-voice mode transparently selects native Standard instrumental.
"""
from __future__ import annotations

import json

from .model_selection import resolve_whisper
from .instrumental_mode import instrumental_aware_style


def build_instrumental_cover(*, source_audio, style, lyrics, ckpt_name, audio_encoder_name,
                             seed, max_duration, whisper_model='auto', whisper_mode='auto',
                             progress_node_id=''):
    from comfy_execution.graph_utils import GraphBuilder
    graph = GraphBuilder()
    model = graph.node('CheckpointLoaderSimple', ckpt_name=ckpt_name)
    encoder = graph.node('AudioEncoderLoader', audio_encoder_name=audio_encoder_name)
    score = graph.node('SheetSage2AudioToABC', audio_encoder=encoder.out(0),
                       audio=source_audio, mode='melody')
    plan = graph.node('YuE2AutoVocalPRO', abc=score.out(0), lyrics=lyrics,
                      max_duration=max_duration, style=style,
                      seed=seed, source_audio=source_audio)
    duration = graph.node('YuE2AdaptiveDurationBudget', abc=plan.out(0),
                          max_duration=float(max_duration))
    conditioning = graph.node('YuE2GenerateMusic', clip=model.out(1), style=instrumental_aware_style(style, lyrics),
                              lyrics=plan.out(1), abc=plan.out(0), seed=int(seed),
                              mode='melody', max_duration=duration.out(0))
    neg = graph.node('ConditioningZeroOut', conditioning=conditioning.out(0))
    lat = graph.node('EmptyYuE2LatentAudio', seconds=conditioning.out(1), batch_size=1)
    sampling = graph.node('KSampler', model=model.out(0), positive=conditioning.out(0),
                          negative=neg.out(0), latent_image=lat.out(0),
                          seed=int(seed), steps=50, cfg=1.0, sampler_name='dpm_2',
                          scheduler='sgm_uniform', denoise=1.0)
    audio = graph.node('VAEDecodeAudio', samples=sampling.out(0), vae=model.out(2))
    report = graph.node('YuE2NativeInstrumentalAudit', audio=audio.out(0),
                         plan_report=plan.out(2), max_duration=float(max_duration),
                         ckpt_name=ckpt_name, audio_encoder_name=audio_encoder_name,
                         whisper_model=whisper_model, whisper_mode=whisper_mode,
                         seed=int(seed), progress_node_id=progress_node_id)
    return {'result': (report.out(0), report.out(1)), 'expand': graph.finalize()}


class YuE2NativeInstrumentalAudit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'audio': ('AUDIO',),
            'plan_report': ('STRING', {'forceInput': True}),
            'max_duration': ('FLOAT', {'default': 240.0, 'min': 1.0, 'max': 900.0}),
            'ckpt_name': ('STRING', {'default': ''}),
            'audio_encoder_name': ('STRING', {'default': ''}),
            'whisper_mode': (['auto', 'selected', 'none'], {'default': 'auto'}),
            'whisper_model': ('STRING', {'default': ''}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'progress_node_id': ('STRING', {'default': ''}),
        }}
    RETURN_TYPES = ('AUDIO', 'STRING')
    RETURN_NAMES = ('AUDIO', 'audit_report')
    FUNCTION = 'audit'
    CATEGORY = 'audio/YuE2'

    def audit(self, audio, plan_report, max_duration, ckpt_name, audio_encoder_name,
              whisper_mode='auto', whisper_model='', seed=0, progress_node_id=''):
        from .reference_progress import report_progress
        from .reference_quality import verify_trimmed_audio
        duration = int(audio['waveform'].shape[-1]) / int(audio['sample_rate'])
        # A small encoder/decoder overshoot is normal; report rather than
        # claiming max_duration is an exact audio length.
        quality = verify_trimmed_audio(audio, minimum_seconds=0)
        text = ''
        asr = {'status': 'not_performed', 'quality_decision': 'unverified'}
        try:
            selected = resolve_whisper(whisper_model, required=False, mode=whisper_mode)
            if selected:
                from .whisper_local import transcribe_selected
                text = transcribe_selected(audio, selected).strip()
                asr = {'status': 'measured_heuristic_only',
                       'quality_decision': 'review_required' if text else 'provisional_no_words',
                       'transcribed_words': text[:400],
                       'notice': 'ASR may hallucinate on music; listening is mandatory'}
        except Exception as exc:
            asr = {'status': 'unavailable', 'quality_decision': 'unverified', 'error': str(exc)}
        try:
            plan = json.loads(plan_report)
        except (TypeError, ValueError):
            plan = {'status': 'invalid_report', 'raw': str(plan_report)[:500]}
        result = {'mode': 'instrumental', 'reference_voice': 'bypassed_by_design',
                  'expected_sung_words': 0, 'plan': plan,
                  'audio_seconds': round(duration, 3), 'max_duration_seconds': max_duration,
                  'duration_over_budget': duration > max_duration * 1.03,
                  'signal_quality': quality,
                  'vocal_asr': asr,
                  'seed': int(seed),
                  'selected_models': {'checkpoint': ckpt_name, 'sheetsage': audio_encoder_name},
                  'audio_verified_instrumental': False,
                  'needs_listening': True}
        if result['duration_over_budget']:
            result['quality_decision'] = 'review_required_duration'
        else:
            result['quality_decision'] = 'review_required' if asr['quality_decision'] == 'review_required' else 'unverified'
        report_progress(progress_node_id, 'execution_completed')
        return audio, json.dumps(result, ensure_ascii=False, indent=2)

NODE_CLASS_MAPPINGS={'YuE2NativeInstrumentalAudit': YuE2NativeInstrumentalAudit}
NODE_DISPLAY_NAME_MAPPINGS={'YuE2NativeInstrumentalAudit': 'YuE2 Native · Instrumental Quality Audit'}
