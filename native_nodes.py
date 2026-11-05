"""Native ComfyUI YuE2 voice-prefix graph nodes, no separate YuE2 runtime."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from .model_store import locations
from .model_selection import resolve_whisper, resolve_mert, resolve_tokenizer, whisper_options, whisper_mode_options
from .native_prefix import encode_with_reference
from .native_plan import YuE2NativeReferencePlan, waveform_mono
from .reference_progress import report_progress



class YuE2NativeSelectReferenceClip:
    """Use one identical reference clip for score, ASR, phonetic prefix and alignment."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'reference_audio':('AUDIO',),
            'reference_lyrics':('STRING',{'default':'','multiline':True}),
            'clip_policy':(['auto','full','limit'],{'default':'auto'}),
            'clip_seconds':('FLOAT',{'default':30.0,'min':15.0,'max':60.0}),
        }}
    RETURN_TYPES=('AUDIO','STRING')
    RETURN_NAMES=('selected_reference_audio','reference_clip_report')
    FUNCTION='select';CATEGORY='audio/YuE2'
    def select(self,reference_audio,reference_lyrics='',clip_policy='auto',clip_seconds=30.0):
        from .reference_strategy import select_reference_clip
        selected,report=select_reference_clip(reference_audio,reference_lyrics,clip_policy,clip_seconds)
        print('[YuE2 Reference Voice][CLIP SELECTION] '+json.dumps(report,ensure_ascii=False))
        return selected,json.dumps(report,ensure_ascii=False)


class YuE2NativeReferenceLyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'reference_audio':('AUDIO',),
                            'reference_lyrics':('STRING',{'multiline':True,'default':''}),
                            'whisper_model':(whisper_options(),{'default':whisper_options()[0]}),
                            'whisper_mode':(whisper_mode_options(),{'default':'auto'}),
                            'progress_node_id':('STRING',{'default':''})}}
    RETURN_TYPES=('STRING','STRING')
    RETURN_NAMES=('lyrics','lyrics_report')
    FUNCTION='resolve';CATEGORY='audio/YuE2'
    def resolve(self,reference_audio,reference_lyrics='',whisper_model='auto',progress_node_id='',whisper_mode='auto'):
        report_progress(progress_node_id,'reference_lyrics_started')
        if reference_lyrics.strip():
            result={'stage':'reference_lyrics','status':'executed','method':'provided_text',
                    'whisper_used':False,'words':len(reference_lyrics.split())}
            print('[YuE2 Reference Voice][LYRICS] '+json.dumps(result,ensure_ascii=False))
            report_progress(progress_node_id,'reference_lyrics_completed')
            return reference_lyrics,json.dumps(result,ensure_ascii=False)
        file=resolve_whisper(whisper_model,required=True,mode=whisper_mode)
        try:
            from .whisper_local import transcribe_selected
            text=transcribe_selected(reference_audio,file)
            if not text:
                raise RuntimeError('Whisper did not detect sung words. Enter reference_lyrics explicitly.')
            result={'stage':'reference_lyrics','status':'executed','method':'whisper',
                    'whisper_used':True,'model_path':str(file),'words':len(text.split())}
            print('[YuE2 Reference Voice][WHISPER] '+json.dumps(result,ensure_ascii=False))
            report_progress(progress_node_id,'reference_lyrics_completed')
            return text,json.dumps(result,ensure_ascii=False)
        except Exception as exc:
            raise RuntimeError('Automatic reference voice transcription failed. Enter reference_lyrics explicitly. '
                               f'Detail: {exc}') from exc


class YuE2NativeReferenceTokens:
    @classmethod
    def INPUT_TYPES(cls):
        from .model_selection import mert_options, tokenizer_options
        merts=mert_options(); toks=tokenizer_options()
        return {'required':{'aligned_reference_audio':('AUDIO',),
                            'mert_model':(merts,{'default':'MERT-v2-FullSong.safetensors' if 'MERT-v2-FullSong.safetensors' in merts else merts[0]}),
                            'tokenizer_name':(toks,{'default':'tokenizer_head_joint_v9.safetensors' if 'tokenizer_head_joint_v9.safetensors' in toks else toks[0]}),
                            'progress_node_id':('STRING',{'default':''})}}
    RETURN_TYPES=('YUE2_PREFIX', 'STRING')
    RETURN_NAMES=('semantic_prefix','codec_report')
    FUNCTION='encode';CATEGORY='audio/YuE2'
    def encode(self,aligned_reference_audio,mert_model='MERT-v2-FullSong.safetensors',tokenizer_name='tokenizer_head_joint_v9.safetensors',progress_node_id=''):
        from .reference_tokenizer import encode_audio
        report_progress(progress_node_id,'mert_tokenizer_started')
        mert_path=resolve_mert(mert_model)
        tokenizer_path=resolve_tokenizer(tokenizer_name)
        codes=encode_audio(aligned_reference_audio,mert_path,tokenizer_path)
        report={'stage':'mert_tokenizer','status':'executed','reference_frames':len(codes),
                'reference_seconds':len(codes)/25,'mert_model':mert_model,
                'mert_path':str(mert_path),'tokenizer_name':tokenizer_name,
                'tokenizer_path':str(tokenizer_path),'tokenizer':'joint-v9 MERT layer-20',
                'runtime':'ComfyUI Python'}
        print('[YuE2 Reference Voice][MERT→TOKENIZER] '+json.dumps(report,ensure_ascii=False))
        report_progress(progress_node_id,'mert_tokenizer_completed')
        return codes,json.dumps(report,ensure_ascii=False)


class YuE2NativeReferenceConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'clip':('CLIP',), 'style':('STRING',{'forceInput':True}),
            'lyrics':('STRING',{'forceInput':True}), 'abc':('STRING',{'forceInput':True}),
            'semantic_prefix':('YUE2_PREFIX',),
            'seed':('INT',{'default':0,'min':0,'max':0xffffffffffffffff}),
            'mode':(['melody','full'],{'default':'full'}),
            'max_duration':('FLOAT',{'default':190.0,'min':1,'max':900}),
            'prefix_seconds':('FLOAT',{'default':0.0,'min':0.0,'max':120.0}),
            'plan_report':('STRING',{'default':''}),
            'budget_policy':(['score_bounded','user_maximum'],{'default':'score_bounded'}),
            'progress_node_id':('STRING',{'default':''}),
        }}
    RETURN_TYPES=('CONDITIONING','FLOAT','STRING','FLOAT')
    RETURN_NAMES=('CONDITIONING','seconds','ar_report','min_target_seconds')
    FUNCTION='run';CATEGORY='audio/YuE2'
    def run(self,clip,style,lyrics,abc,semantic_prefix,seed,mode,max_duration,prefix_seconds=0.0,progress_node_id='',plan_report='',budget_policy='score_bounded'):
        # max_duration is the requested *output song* budget, not the composite
        # (reference prompt + target song). The reference is trimmed downstream.
        if prefix_seconds<0 or max_duration<=0:
            raise ValueError('Invalid reference/target duration')
        from .reference_quality import nominal_target_seconds, minimum_target_seconds
        from .reference_strategy import reference_generation_budget
        target_nominal = nominal_target_seconds(plan_report)
        target_budget, budget_report = reference_generation_budget(target_nominal, max_duration, policy=budget_policy)
        combined_duration=target_budget+float(prefix_seconds)
        min_target = minimum_target_seconds(target_nominal, target_budget)
        report_progress(progress_node_id,'ar_started')
        # YuE2's melody cover mode expects a chord-free score. Keep the
        # verified composite score intact; remove only score chord annotations
        # from the CLIP payload after proving that notes/meters/keys agree.
        from .reference_strategy import score_for_conditioning
        conditioning_abc, score_encoding = score_for_conditioning(abc, mode)
        # Observe the exact arguments supplied to the unchanged native prefix
        # adapter. No mutation of text, ABC, semantic tokens or sampling logic.
        from .reference_diagnostics import sha256_text
        diagnostic_trace={
            'observed_before_clip_tokenize':True,
            'lyrics_sha256':sha256_text(lyrics),
            'abc_sha256':sha256_text(conditioning_abc),
            'planned_composite_abc_sha256':sha256_text(abc),
            'conditioning_score_preparation':score_encoding,
            'style_sha256':sha256_text(style),
            'lyrics_chars':len(lyrics), 'abc_chars':len(conditioning_abc),
            'style_chars':len(style), 'sampling_mode':mode,
            'configured_max_duration_with_prefix_s':combined_duration,
            'reference_target_budget':budget_report,
            'reference_codec_frames_requested':len(semantic_prefix),
        }
        conditioning,seconds=encode_with_reference(clip,style,lyrics,conditioning_abc,semantic_prefix,seed,mode,combined_duration,
                                                 progress_node_id=progress_node_id,min_target_seconds=min_target)
        # Native _generate signals budget exhaustion in conditioning metadata.
        # This previously disappeared before the execution report, causing a
        # truncated target to be reported simply as successful generation.
        try:
            semantic_truncated = conditioning[0][1].get('yue2_truncated') is True
        except (TypeError, KeyError, IndexError, AttributeError):
            semantic_truncated = None  # Test double or different native layout.
        try:
            expected_lyrics_hash=json.loads(plan_report).get('diagnostic_lexical_inputs', {}).get('composite', {}).get('raw_sha256')
        except (ValueError,TypeError,AttributeError):
            expected_lyrics_hash=None
        diagnostic_trace['plan_composite_hash_matches_encoder'] = (
            sha256_text(lyrics) == expected_lyrics_hash if expected_lyrics_hash else None)
        from .reference_quality import validate_semantic_duration
        validate_semantic_duration(int(round(float(seconds) * 25)), len(semantic_prefix), min_target)
        report_progress(progress_node_id,'ar_completed')
        report={'stage':'ar_prefix','status':'executed','prefix_injected':True,
                'prefix_tokens':len(semantic_prefix),'generated_duration_seconds':seconds,
                'native_yue2_clip_used':True,'mode':mode,
                'target_duration_budget_seconds':float(max_duration),
                'actual_target_semantic_ceiling_s':target_budget,
                'budget_contract':budget_report,
                'reference_prefix_seconds':float(prefix_seconds),
                'combined_duration_budget_seconds':combined_duration,
                'target_score_nominal_seconds':target_nominal,
                'minimum_accepted_target_seconds':min_target,
                'actual_target_seconds':float(seconds)-float(prefix_seconds),
                'semantic_truncated':semantic_truncated,
                'diagnostic_encoder_payload':diagnostic_trace}
        print('[YuE2 Reference Voice][AR] '+json.dumps(report,ensure_ascii=False))
        return conditioning,seconds,json.dumps(report,ensure_ascii=False),min_target


class YuE2NativeReferenceTracePlan:
    """Observe the already-composed reference/target text, without changing it."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'plan_report':('STRING',{'forceInput':True}),
            'composite_lyrics':('STRING',{'forceInput':True}),
            'reference_lyrics':('STRING',{'forceInput':True}),
            'target_lyrics':('STRING',{'forceInput':True}),
        }}
    RETURN_TYPES=('STRING',)
    RETURN_NAMES=('diagnostic_plan_report',)
    FUNCTION='trace';CATEGORY='audio/YuE2'
    def trace(self,plan_report,composite_lyrics,reference_lyrics,target_lyrics):
        from .reference_diagnostics import composition_proof
        evidence=json.loads(plan_report)
        evidence['diagnostic_lexical_inputs']=composition_proof(
            reference_lyrics,target_lyrics,composite_lyrics)
        if not evidence['diagnostic_lexical_inputs']['all_words_in_order']:
            raise RuntimeError('Reference Voice diagnostic detected a lexical boundary violation')
        return (json.dumps(evidence,ensure_ascii=False),)


class YuE2NativeProgressLatent:
    """Pass-through for *real* KSampler start/end, routed to wrapper progress."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'latent':('LATENT',),
                            'stage':(['nar_sampling_started','nar_sampling_completed'],),
                            'progress_node_id':('STRING',{'default':''})}}
    RETURN_TYPES=('LATENT',)
    FUNCTION='pass_through'; CATEGORY='audio/YuE2'
    def pass_through(self,latent,stage,progress_node_id=''):
        report_progress(progress_node_id,stage)
        return (latent,)


class YuE2NativeTrimReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'audio':('AUDIO',), 'prefix_seconds':('FLOAT',{'forceInput':True}),
                            'progress_node_id':('STRING',{'default':''}),
                            'min_output_seconds':('FLOAT',{'default':0.0,'min':0.0,'max':900.0})}}
    RETURN_TYPES=('AUDIO',)
    FUNCTION='trim';CATEGORY='audio/YuE2'
    def trim(self,audio,prefix_seconds,progress_node_id='',min_output_seconds=0.0):
        report_progress(progress_node_id,'audio_decoded')
        sr=int(audio['sample_rate']); t=audio['waveform']
        cut=int(round(float(prefix_seconds)*sr))
        if t.ndim!=3 or cut<0 or cut>=t.shape[-1]:
            raise RuntimeError('Generated reference-conditioned song has no audio after prefix; increase duration')
        from .reference_quality import verify_trimmed_audio
        verify_trimmed_audio({'waveform': t[...,cut:], 'sample_rate': sr}, minimum_seconds=min_output_seconds)
        report_progress(progress_node_id,'prefix_trimmed')
        return ({'waveform':t[...,cut:].contiguous(),'sample_rate':sr},)


class YuE2NativeExecutionReport:
    """Execution evidence from nodes that actually completed, not a readiness report.

    The AUDIO input is downstream of the NAR LoRA -> sampler -> decoder ->
    reference prefix trim. Returning the report requires that entire path to
    resolve successfully. The report does not claim measured voice similarity.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'audio':('AUDIO',), 'plan_report':('STRING',{'forceInput':True}),
            'lyrics_report':('STRING',{'forceInput':True}),
            'tokenizer_report':('STRING',{'forceInput':True}),
            'ar_report':('STRING',{'forceInput':True}),
            'ckpt_name':('STRING',{'default':''}),
            'audio_encoder_name':('STRING',{'default':''}),
            'mert_model':('STRING',{'default':''}),
            'tokenizer_name':('STRING',{'default':''}),
            'nar_lora_name':('STRING',{'default':''}),
            'whisper_model':('STRING',{'default':'auto'}),
            'whisper_mode':('STRING',{'default':'auto'}),
            'target_lyrics':('STRING',{'default':''}),
            'reference_text':('STRING',{'default':''}),
            'source_contamination_audit':('BOOLEAN',{'default':False}),
            'seed':('INT',{'default':0,'min':0,'max':0xffffffffffffffff}),
            'model_paths':('STRING',{'default':''}),
            'progress_node_id':('STRING',{'default':''}),
            'diagnostic_mode':(['basic','segmented','off'],{'default':'basic'}),
            'diagnostics_include_text':('BOOLEAN',{'default':False}),
        }, 'optional':{'source_audio':('AUDIO',),'reference_audio':('AUDIO',),
                        'reference_clip_report':('STRING',{'default':''}),
                        'reference_style_report':('STRING',{'default':''}),
                        'reference_quality_gate':(['report_only','fail_on_strong_reference_leakage'],{'default':'report_only'})}}
    RETURN_TYPES=('AUDIO','STRING')
    RETURN_NAMES=('AUDIO','execution_report')
    FUNCTION='summarize';CATEGORY='audio/YuE2'
    def summarize(self,audio,plan_report,lyrics_report,tokenizer_report,ar_report,
                  ckpt_name,audio_encoder_name,mert_model,tokenizer_name,
                  nar_lora_name,whisper_model,seed,model_paths="",progress_node_id="",whisper_mode="auto",target_lyrics="",reference_text="",source_audio=None,source_contamination_audit=False,
                  reference_audio=None,diagnostic_mode='basic',diagnostics_include_text=False,
                  reference_clip_report='',reference_style_report='',reference_quality_gate='report_only'):
        if not isinstance(audio,dict) or 'waveform' not in audio or 'sample_rate' not in audio:
            raise RuntimeError('Reference Voice decoder returned an invalid AUDIO object')
        sample_rate=int(audio['sample_rate'])
        samples=int(audio['waveform'].shape[-1])
        if samples<=0 or sample_rate<=0:
            raise RuntimeError('Reference Voice decoder produced empty audio')
        evidence={}
        for key,value in [('plan',plan_report),('reference_lyrics',lyrics_report),
                          ('mert_tokenizer',tokenizer_report),('ar_prefix',ar_report)]:
            try:
                evidence[key]=json.loads(value)
            except (ValueError,TypeError) as exc:
                raise RuntimeError(f'Invalid execution evidence from {key}') from exc
        if evidence['mert_tokenizer'].get('status')!='executed' or \
           evidence['ar_prefix'].get('prefix_injected') is not True:
            raise RuntimeError('Reference Voice model execution/injection evidence is incomplete')
        try:
            resolved=json.loads(model_paths) if model_paths else {}
        except (ValueError,TypeError):
            resolved={}
        from .reference_quality import verify_trimmed_audio
        quality = verify_trimmed_audio(audio, minimum_seconds=float(evidence['ar_prefix'].get('minimum_accepted_target_seconds', 0)))
        if diagnostic_mode not in ('off','basic','segmented'):
            raise ValueError('Unknown Reference Voice diagnostic mode: '+str(diagnostic_mode))
        diagnostic_segments = {'status':'disabled' if diagnostic_mode != 'segmented' else 'unavailable', 'windows':[]}
        lyric_audit = {'status': 'not_performed', 'reason': 'No target Lyrics supplied'}
        if target_lyrics and str(target_lyrics).strip():
            lyric_audit = {'status': 'not_performed', 'reason': 'Whisper mode disabled or no installed model'}
            selected_whisper = resolved.get('whisper')
            if selected_whisper:
                try:
                    from .whisper_local import transcribe_selected
                    from .lyrics_audit import compare_lyrics
                    if diagnostic_mode == 'segmented':
                        from .whisper_local import transcribe_selected_windows
                        from .reference_diagnostics import evidence_for_segment
                        windows = transcribe_selected_windows(audio, selected_whisper)
                        hypothesis = ' '.join(w['text'] for w in windows)
                        diagnostic_segments = {
                            'status':'measured_heuristic_only',
                            'asr_input':'trimmed_reference_voice_output',
                            'model':str(selected_whisper),
                            'windows':[{**{'start_s':w['start_s'],'end_s':w['end_s']},
                                **evidence_for_segment(w['text'],target_lyrics,reference_text,
                                                      include_text=diagnostics_include_text)} for w in windows],
                        }
                    else:
                        hypothesis = transcribe_selected(audio, selected_whisper)
                    source_transcript = ''
                    source_audit_status = 'disabled'
                    if source_contamination_audit and source_audio is not None:
                        try:
                            source_transcript = transcribe_selected(source_audio, selected_whisper)
                            source_audit_status = 'measured_heuristic_only'
                        except Exception as source_exc:
                            source_audit_status = 'unavailable: '+str(source_exc)
                    lyric_audit = compare_lyrics(target_lyrics, hypothesis,
                                                 reference_text=reference_text,
                                                 source_text=source_transcript)
                    lyric_audit['source_asr_status'] = source_audit_status
                    lyric_audit['status'] = 'measured_heuristic_only'
                except Exception as exc:
                    lyric_audit = {'status': 'unavailable', 'reason': str(exc)}
        effective_quality = lyric_audit.get('quality_decision', 'unverified')
        execution_state = 'completed_with_lyrics_review_required' if effective_quality == 'review_required' else 'completed_audio_not_lexically_certified'
        if evidence['ar_prefix'].get('semantic_truncated') is True:
            # The sequence hit the sampler token ceiling without MUSIC_END;
            # the audio may be usable but is not a naturally ended song.
            execution_state='completed_with_semantic_truncation_and_lyrics_review_required'
        results={'execution':execution_state,'not_only_inventory':True,
            'semantic_end_reason':('token_budget_exhausted' if evidence['ar_prefix'].get('semantic_truncated') is True
                                   else 'native_end_or_unavailable'),
            'lyrics_quality_decision':effective_quality,
            'audio_lexical_fidelity_certified':False,
            'resolved_model_paths':resolved,
            'selected_models':{'yue2_checkpoint':ckpt_name,'sheetsage':audio_encoder_name,
                'mert':mert_model,'tokenizer':tokenizer_name,
                'nar_lora':nar_lora_name,'whisper_selection':whisper_model,'whisper_mode':whisper_mode},
            'stages':{'reference_lyrics':evidence['reference_lyrics'],
                'score_alignment':evidence['plan'],
                'mert_and_tokenizer':evidence['mert_tokenizer'],
                'ar_prefix':evidence['ar_prefix'],
                'nar_lora_sampler_vae_trim':{'status':'graph_completed',
                    'reason':'audio depends on selected LoraLoaderModelOnly, KSampler, VAEDecodeAudio and TrimReference',
                    'lora_effect_measured':False}},
            'output_audio':{'seconds':round(samples/sample_rate,3),
                            'sample_rate':sample_rate,'sample_count':samples, 'signal_quality': quality},
            'lyrics_fidelity_audit': lyric_audit,
            'seed':int(seed),
            'progress':{'type':'pipeline_milestones','parent_node_id':str(progress_node_id),
                        'percent':100,'sampler_steps':64,
                        'note':'Milestones indicate completed phases, not remaining-time estimates'},
            'limitations':'Graph execution is verified. Voice similarity, singing quality and the effect size of the LoRA require listening/measurement on your GPU.'}
        # Policy and leakage evidence are settled BEFORE the diagnostic ZIP is
        # generated; a strict quality gate still exports a failure report.
        if reference_quality_gate not in ('report_only','fail_on_strong_reference_leakage'):
            raise ValueError('Unknown reference_quality_gate: '+str(reference_quality_gate))
        try:
            selected_clip=json.loads(reference_clip_report) if reference_clip_report else {}
            style_strategy=json.loads(reference_style_report) if reference_style_report else {}
        except (ValueError,TypeError) as exc:
            raise ValueError('Malformed Reference Voice strategy report') from exc
        windows=diagnostic_segments.get('windows',[])
        exclusive_ref=sum(w.get('reference_exclusive_hits',0) for w in windows)
        exclusive_target=sum(w.get('target_exclusive_hits',0) for w in windows)
        strong_leakage=(len(windows)>=2 and exclusive_ref>=8 and exclusive_target==0
                        and sum(w.get('recognized_words',0) for w in windows)>=12)
        results['reference_strategy']={
            'clip':selected_clip, 'style':style_strategy,
            'quality_gate':reference_quality_gate,
            'conditioning_mode':evidence['ar_prefix'].get('mode'),
            'target_semantic_ceiling_s':evidence['ar_prefix'].get('actual_target_semantic_ceiling_s'),
            'reference_exclusive_hits':exclusive_ref,
            'target_exclusive_hits':exclusive_target,
            'strong_reference_lexical_signal':strong_leakage,
        }
        if strong_leakage:
            results['execution']='completed_reference_lyrics_contamination_suspected'
            results['lyrics_quality_decision']='reference_lyrics_contamination_suspected'
        if diagnostic_mode != 'off':
            from .reference_diagnostics import (audio_fingerprint, compact_report,
                diagnosis, export_bundle)
            results['diagnostic_audio_roles']={
                'source':audio_fingerprint(source_audio),
                'reference':audio_fingerprint(reference_audio),
                'generated_target':audio_fingerprint(audio)}
            results['diagnostic_segment_asr']=diagnostic_segments
            trace=compact_report(results, include_text=diagnostics_include_text)
            findings=diagnosis(trace)
            results['diagnostic_findings']=findings
            trace['diagnosis']=findings
            results['diagnostic_bundle']=export_bundle(trace)
            print('[YuE2 Reference Voice][DIAGNOSTIC BUNDLE] '+
                  json.dumps(results['diagnostic_bundle'],ensure_ascii=False))
        if strong_leakage and reference_quality_gate=='fail_on_strong_reference_leakage':
            raise RuntimeError('Reference Voice output is strongly reference-like according to '
                'segmented Whisper: target-exclusive hits=0, reference-exclusive '
                f'hits={exclusive_ref}. Audio rejected by requested quality gate; diagnostic '
                'ZIP was exported. This is a heuristic measure, not acoustic certainty.')
        summary=json.dumps(results,ensure_ascii=False,indent=2,default=str)
        print('[YuE2 Reference Voice][EXECUTION COMPLETED] '+json.dumps({
            'execution':results['execution'],'quality_decision':results['lyrics_quality_decision'],
            'diagnostic_bundle':results.get('diagnostic_bundle'),
            'output_seconds':round(samples/sample_rate,3)},ensure_ascii=False))
        report_progress(progress_node_id,'execution_completed')
        return audio,summary


NODE_CLASS_MAPPINGS={x.__name__:x for x in (
    YuE2NativeSelectReferenceClip,YuE2NativeReferenceLyrics,YuE2NativeReferencePlan,
    YuE2NativeReferenceTokens,YuE2NativeReferenceConditioning,YuE2NativeReferenceTracePlan,YuE2NativeProgressLatent,
    YuE2NativeTrimReference,
    YuE2NativeExecutionReport)}
NODE_DISPLAY_NAME_MAPPINGS={
    'YuE2NativeSelectReferenceClip':'YuE2 Native · Select Aligned Reference Clip',
    'YuE2NativeReferenceLyrics':'YuE2 Native · Reference Lyrics',
    'YuE2NativeValidateGeneratedScore':'YuE2 Native · Validate Generated Score',
    'YuE2NativeReferencePlan':'YuE2 Native · Composite Score / Alignment',
    'YuE2NativeReferenceTracePlan':'YuE2 Native · Reference/Target Lexical Trace',
    'YuE2NativeReferenceTokens':'YuE2 Native · Encode Vocal Reference',
    'YuE2NativeReferenceConditioning':'YuE2 Native · AR Voice Prefix Conditioning',
    'YuE2NativeProgressLatent':'YuE2 Native · NAR Progress Marker',
    'YuE2NativeTrimReference':'YuE2 Native · Remove Reference Intro',
    'YuE2NativeExecutionReport':'YuE2 Native · Reference Voice Execution Audit',
}


class YuE2NativeValidateGeneratedScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'generated_abc': ('STRING', {'forceInput': True}),
            'source_abc': ('STRING', {'forceInput': True}),
            'lyrics': ('STRING', {'forceInput': True}),
            'source_report': ('STRING', {'default': ''}),
            'reason': ('STRING', {'default': ''}),
            'progress_node_id': ('STRING', {'default': ''}),
            'reference_composite': ('BOOLEAN', {'default': False}),
            'max_duration': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 900.0}),
            'vocal_melody_mode': (['Adaptive', 'Music adapts to Lyrics', 'Lyrics adapt to Music'], {'default':'Adaptive'}),
        }, 'optional': {
            'source_audio': ('AUDIO',),
            'section_policy': (['all_of_rest','verses_and_choruses','all_sections'], {'default':'all_of_rest'}),
            'instrumental_handling': (['replace_with_rests','keep_original'], {'default':'replace_with_rests'}),
            'existing_vocal': (['preserve','overwrite'], {'default':'preserve'}),
            'instrumental_tags': ('STRING',{'default':''}),
            'vocal_tags': ('STRING',{'default':''}),
        }}
    RETURN_TYPES = ('STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('abc', 'lyrics', 'auto_score_report')
    FUNCTION = 'validate'
    CATEGORY = 'audio/YuE2'

    def validate(self, generated_abc, source_abc, lyrics, source_report='', reason='', progress_node_id='', reference_composite=False, max_duration=0.0, source_audio=None, vocal_melody_mode='Adaptive', section_policy='all_of_rest', instrumental_handling='replace_with_rests', existing_vocal='preserve', instrumental_tags='', vocal_tags=''):
        """Validate a generated *musical* candidate with the SAME production
        contract as SheetSage: native score, real duration, exact target words.

        No retry through `optimize` and no new auto-ABC generation loop.
        """
        from .cover_contract import prepare_cover
        from .engine import convert_abc
        from .lyric_contract import _tokens
        from .reference_progress import report_progress
        errors=[]
        for name, raw in (('source_score_strict_fallback',source_abc),
                          ('official_YuE2GenerateABC',generated_abc)):
            try:
                converted, _, conv_report = convert_abc(raw, lyrics,
                    section_policy=section_policy,
                    instrumental_handling=instrumental_handling,
                    existing_vocal=existing_vocal,
                    instrumental_tags=instrumental_tags,
                    vocal_tags=vocal_tags, strict=False)
                plan = prepare_cover(converted, lyrics, max_duration=max_duration, origin=name)
                report=dict(plan.report)
                report['score_source']=(name.replace('_strict_fallback', '_bounded_musical_excerpt')
                    if name.startswith('source_score') and report['score_durations']['excerpt'] else name)
                report['vocal_melody_alignment'] = {
                    'word_order_unchanged': True,
                    'action':report['score_durations']['action'],
                    'recovery':report['score_durations'].get('source_trace'),
                    'score_changed':report['score_durations']['excerpt']}
                report['vocal_melody_mode']=vocal_melody_mode
                report['generated_abc_validation']='same_canonical_cover_contract'
                report['source_report']=source_report if name=='source_score_fallback' else None
                report['fitness']=plan.report['fitness_advisory']
                report['original_upload_melody_preserved']=(name.startswith('source_score'))
                report['warnings']=report.get('warnings',[])+errors
                if _tokens(plan.lyrics)[0] != _tokens(lyrics)[0]:
                    raise ValueError('Target Lyrics word sequence changed')
                report_progress(progress_node_id, 'score_selection_completed')
                print('[YuE2 Reference Voice][SCORE] '+json.dumps({
                    'decision':report['decision'], 'score_source':name,
                    'duration':report['score_durations']['selected_seconds'],
                    'lyrics_words':len(_tokens(lyrics)[0]),
                    'empty_vocal_sections':report['section_contract']['empty_vocal_sections']},
                    ensure_ascii=False))
                from .lyric_contract import with_terminal_end
                return plan.abc, with_terminal_end(plan.lyrics), json.dumps(report,ensure_ascii=False)
            except ValueError as exc:
                errors.append(name+': '+str(exc))
        raise ValueError('Neither official generated nor source ABC can be used: '+'; '.join(errors))

NODE_CLASS_MAPPINGS.update({YuE2NativeValidateGeneratedScore.__name__: YuE2NativeValidateGeneratedScore})
