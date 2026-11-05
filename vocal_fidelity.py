"""Post-decode output audit: identical lyric authority for Standard covers.

ASR on singing is fallible. This node never declares *guaranteed* fidelity;
report_only leaves the waveform available for listening while fail_closed
requires ASR evidence and rejects clearly non-matching results.
"""
from __future__ import annotations
import json
from .instrumental_mode import instrumental_requested
from .model_selection import whisper_mode_options, whisper_options, resolve_whisper
from .lyrics_audit import compare_lyrics, word_tokens


class YuE2VocalFidelityAudit:
    @classmethod
    def INPUT_TYPES(cls):
        models=whisper_options()
        return {'required': {
            'audio': ('AUDIO',),
            'lyrics': ('STRING', {'forceInput': True}),
            'max_duration': ('FLOAT', {'default': 240.0, 'min': 0.0, 'max': 900.0}),
            'whisper_mode': (whisper_mode_options(), {'default': 'auto'}),
            'whisper_model': (models, {'default': models[0]}),
            'quality_gate': (['report_only', 'fail_closed'], {'default': 'report_only'}),
            'source_contamination_audit': ('BOOLEAN', {'default': False}),
        }, 'optional': {
            'source_audio': ('AUDIO',),
            'reference_text': ('STRING', {'default': ''}),
            'selected_mode_report': ('STRING', {'default': '', 'forceInput': True}),
        }}
    RETURN_TYPES = ('AUDIO', 'STRING')
    RETURN_NAMES = ('AUDIO', 'quality_report')
    FUNCTION = 'audit'
    CATEGORY = 'audio/YuE2'
    DESCRIPTION = 'Audit decoded audio and immutable user Lyrics. Singing ASR is imperfect; fail_closed is opt-in.'

    def audit(self, audio, lyrics, max_duration=0.0, whisper_mode='auto', whisper_model='',
              quality_gate='report_only', source_contamination_audit=False,
              source_audio=None, reference_text='', selected_mode_report=''):
        if quality_gate not in ('report_only','fail_closed'):
            raise ValueError('Unknown quality gate: '+quality_gate)
        # Always inspect the exact selected waveform before SaveAudio.
        # A silent AUDIO tensor is a failed generation regardless of whether
        # ASR is enabled or quality_gate is report_only.
        from .audio_signal_quality import measure_audio_signal_quality
        signal_quality = measure_audio_signal_quality(audio)
        duration = float(signal_quality['duration_seconds'])
        instrumental=instrumental_requested(lyrics)
        from .lyric_contract import _tokens
        output={'target_mode':'instrumental' if instrumental else 'sung',
                'requested_words':0 if instrumental else len(_tokens(lyrics)[0]),
                'audio_seconds':round(duration,3),
                'signal_quality':signal_quality,
                'max_duration_seconds':float(max_duration),
                'duration_over_budget':max_duration>0 and duration>max_duration*1.03,
                'quality_gate':quality_gate,
                'audio_guaranteed_lexically_faithful':False,
                'asr':{'status':'not_available','quality_decision':'unverified'},
                'quality_decision':'unverified','requires_listening':True}
        # This report MUST describe the waveform actually delivered to
        # SaveAudio, after the Standard / Reference Voice selector. Older
        # workflows wired the Standard path directly into this node and thus
        # displayed a plausible but wrong result in Reference Voice mode.
        mode = 'unknown'
        if selected_mode_report:
            try:
                selection = json.loads(selected_mode_report)
                if isinstance(selection,dict):
                    mode = selection.get('selected_mode') or (
                        'reference_voice' if 'ar_prefix' in selection.get('stages',{})
                        or str(selection.get('execution','')).startswith('completed_') else 'unknown')
                    output['selected_path_report_available'] = True
            except (ValueError,TypeError):
                output['selected_path_report_invalid'] = True
        output['audited_audio_path'] = mode
        output['audio_path_matches_selected_output'] = bool(selected_mode_report)
        selected=None
        try:
            selected=resolve_whisper(whisper_model,required=quality_gate=='fail_closed',mode=whisper_mode)
        except Exception as exc:
            output['asr']={'status':'unavailable','reason':str(exc),'quality_decision':'unverified'}
        if selected:
            try:
                from .whisper_local import transcribe_selected
                transcript=transcribe_selected(audio,selected)
                if instrumental:
                    tokens=word_tokens(transcript)
                    output['asr']={'status':'measured_heuristic_only',
                        'potential_unrequested_words':tokens[:30],
                        'suspected_singing_words':len(tokens),
                        'quality_decision':'review_required' if len(tokens)>=3 else 'provisional_no_words'}
                else:
                    source_text=''
                    if source_contamination_audit and source_audio is not None:
                        try: source_text=transcribe_selected(source_audio,selected)
                        except Exception as exc: output['source_transcription_error']=str(exc)
                    output['asr']=compare_lyrics(lyrics,transcript,reference_text=reference_text,source_text=source_text)
                output['whisper_model_used']=str(selected)
            except Exception as exc:
                output['asr']={'status':'unavailable','reason':str(exc),'quality_decision':'unverified'}
        # Word recognition is NOT the same thing as phonetic clarity.
        # Record low alignment as a listening priority, not a diagnosis:
        # ASR can hallucinate, miss sung words, or be masked by the music.
        coverage = output['asr'].get('expected_word_coverage_proxy')
        if not instrumental and isinstance(coverage, (float, int)):
            output['diction_review'] = {
                'status': 'listen_to_verify',
                'word_order_coverage_proxy': round(float(coverage), 4),
                'possible_contributors': ['unclear_enunciation', 'different_sung_words',
                                          'phrase_timing', 'backing_music_masking', 'ASR_error'],
                'cause_verified': False,
            }
        else:
            output['diction_review'] = {'status': 'not_measured', 'cause_verified': False}
        report_decision=output['asr'].get('quality_decision','unverified')
        if signal_quality.get('quality_warning') == 'mostly_silent':
            output['quality_decision'] = 'review_required_mostly_silent'
        elif output['duration_over_budget']:
            output['quality_decision']='review_required_duration'
        elif report_decision=='review_required':
            output['quality_decision']='review_required_lyrics'
        elif report_decision in ('provisional_pass','provisional_no_words'):
            output['quality_decision']='provisional_pass_requires_listening'
        if quality_gate=='fail_closed' and output['quality_decision'] not in ('provisional_pass_requires_listening',):
            raise RuntimeError('YuE2 Vocal Fidelity gate rejected/unverified generation: '+json.dumps(output,ensure_ascii=False))
        print('[YuE2 Auto Vocal PRO][POST AUDIO AUDIT] '+json.dumps(output,ensure_ascii=False))
        return audio, json.dumps(output, ensure_ascii=False, indent=2)

NODE_CLASS_MAPPINGS={'YuE2VocalFidelityAudit': YuE2VocalFidelityAudit}
NODE_DISPLAY_NAME_MAPPINGS={'YuE2VocalFidelityAudit': 'YuE2 · Final Audio / Lyrics Audit'}
