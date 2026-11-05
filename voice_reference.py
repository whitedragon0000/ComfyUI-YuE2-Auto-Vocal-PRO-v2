"""ComfyUI-native reference-voice cover, built as an internal ComfyUI graph.

All inference (native YuE2 CLIP/DiT/VAE, native SheetSage2, tokenization) is
performed inside ComfyUI Python. No yue2-infer, WSL, external service, or
second YuE2 checkpoint format is required.
"""
from __future__ import annotations
import json
from .model_store import NATIVE_SAGE
from .reference_progress import report_progress, start_run
from .model_selection import (
    checkpoint_options, sheetsage_options, mert_options, tokenizer_options,
    lora_options, whisper_options, whisper_mode_options, validate_reference_models,
)




class YuE2ReferenceVoiceCover:
    @classmethod
    def INPUT_TYPES(cls):
        options=checkpoint_options()
        encoders=sheetsage_options()
        merts=mert_options()
        tokenizers=tokenizer_options()
        loras=lora_options()
        whispers=whisper_options()
        return {'hidden':{'unique_id':'UNIQUE_ID'},'required':{
            'source_audio':('AUDIO',),
            'reference_audio':('AUDIO',),
            'style':('STRING',{'forceInput':True}),
            'lyrics':('STRING',{'forceInput':True}),
            'reference_lyrics':('STRING',{'multiline':True,'default':'',
               'tooltip':'Lyrics sung in the reference sample; blank uses optional Whisper model.'}),
            'first_note_seconds':('FLOAT',{'default':-1.0,'min':-1.0,'max':60.0,'step':0.01,
               'tooltip':'-1 = estimate first note from reference audio; manual value improves precision.'}),
            'ckpt_name':(options,{'default':'yue2_3b_bf16.safetensors' if 'yue2_3b_bf16.safetensors' in options else options[0]}),
            'audio_encoder_name':(encoders,{'default':NATIVE_SAGE if NATIVE_SAGE in encoders else encoders[0]}),
            'seed':('INT',{'default':0,'min':0,'max':0xffffffffffffffff}),
            'max_duration':('FLOAT',{'default':240.0,'min':1.0,'max':900.0}),
            'mert_model':(merts,{'default':'MERT-v2-FullSong.safetensors' if 'MERT-v2-FullSong.safetensors' in merts else merts[0]}),
            'tokenizer_name':(tokenizers,{'default':'tokenizer_head_joint_v9.safetensors' if 'tokenizer_head_joint_v9.safetensors' in tokenizers else tokenizers[0]}),
            'nar_lora_name':(loras,{'default':'nar_lora_joint_v9_comfyui.safetensors' if 'nar_lora_joint_v9_comfyui.safetensors' in loras else loras[0]}),
            'whisper_model':(whispers,{'default':'whisper-large-v3-turbo.safetensors' if 'whisper-large-v3-turbo.safetensors' in whispers else whispers[0], 'tooltip':'File realmente presente in ComfyUI/models/whisper; usato soltanto con modalita selected.'}),
            'whisper_mode':(whisper_mode_options(),{'default':'auto', 'tooltip':'auto: usa il modello preferito disponibile; selected: usa il file scelto; none: disabilita ASR. Le Lyrics di riferimento manuali hanno sempre precedenza.'}),
            'source_contamination_audit':('BOOLEAN',{'default':False,'tooltip':'Also transcribe source song for optional ASR-based contamination diagnostic (slow).'}),
            'diagnostic_mode':(['basic','segmented','off'],{'default':'basic',
                'tooltip':'basic: export privacy-safe ZIP trace; segmented: single-load windowed Whisper on output; off: no diagnostic ZIP.'}),
            'diagnostics_include_text':('BOOLEAN',{'default':False,
                'tooltip':'Optional ASR token excerpts in diagnostic ZIP. OFF keeps all actual lyrics/transcripts private; hashes/counts still present.'}),
            'reference_clip_policy':(['auto','full','limit'],{'default':'auto',
                'tooltip':'auto: use ~30s only when reference lyrics come from Whisper. With manual reference lyrics retain entire clip; full: always retain; limit: explicit clip reduction.'}),
            'reference_clip_seconds':('FLOAT',{'default':30.0,'min':15.0,'max':60.0,'step':1.0,
                'tooltip':'Limit reference input before ALL reference ASR, score and tokenization (15-60 seconds). A lower-energy boundary is chosen.'}),
            'conditioning_mode':(['full','melody'],{'default':'full',
                'tooltip':'melody: official YuE2 recommendation for covers; full: restore previous Reference Voice composition for A/B.'}),
            'target_language':(['auto','Italian','English','Spanish','French','German','Portuguese','Chinese','Japanese','Korean'],{'default':'auto',
                'tooltip':'Guide vocal language in Style; does not transcribe or replace target Lyrics.'}),
            'reference_quality_gate':(['report_only','fail_on_strong_reference_leakage'],{'default':'report_only',
                'tooltip':'Fail only if segmented Whisper strongly favors reference-exclusive words over target-exclusive words. ASR is heuristic.'}),
            'legacy_prefix_behavior':('BOOLEAN',{'default':False,
                'tooltip':'A/B baseline: restore full 40-45 s reference, original blank Style, full conditioning and user maximum AR. Overrides the controlled strategy settings.'}),
            'forensic_replay':('BOOLEAN',{'default':False,
                'tooltip':'INCLUDE ABC, Lyrics and reports in clear text for offline reproducibility. No audio is exported. Contains private/copyright text.'}),
        }}
    RETURN_TYPES=('AUDIO','STRING')
    RETURN_NAMES=('AUDIO','report')
    FUNCTION='generate';CATEGORY='audio/YuE2'
    DESCRIPTION='Native ComfyUI voice-prefix cover using the installed YuE2 checkpoint and ComfyUI NAR LoRA; never launches a second runtime.'

    def generate(self,source_audio,reference_audio,style,lyrics,reference_lyrics='',
                 first_note_seconds=-1.0,ckpt_name='yue2_3b_bf16.safetensors',
                 audio_encoder_name=NATIVE_SAGE,seed=0,max_duration=240.0,
                 mert_model='MERT-v2-FullSong.safetensors',
                 tokenizer_name='tokenizer_head_joint_v9.safetensors',
                 nar_lora_name='nar_lora_joint_v9_comfyui.safetensors',
                 whisper_model='auto',whisper_mode='auto',unique_id=None,source_contamination_audit=False,
                 diagnostic_mode='basic',diagnostics_include_text=False,
                 reference_clip_policy='auto',reference_clip_seconds=30.0,
                 conditioning_mode='full',target_language='auto',reference_quality_gate='report_only',
                 legacy_prefix_behavior=False, forensic_replay=False):
        progress_node_id=start_run(unique_id)
        from .forensic_recorder import attach_progress
        attach_progress(progress_node_id)
        report_progress(progress_node_id,'graph_prepared')
        from .instrumental_mode import instrumental_requested
        if instrumental_requested(lyrics):
            from .instrumental_graph import build_instrumental_cover
            print('[YuE2 Reference Voice] Instrumental request: bypassing singer prefix and preserving Standard instrumental generator')
            return build_instrumental_cover(
                source_audio=source_audio, style=style, lyrics=lyrics,
                ckpt_name=ckpt_name, audio_encoder_name=audio_encoder_name,
                seed=seed, max_duration=max_duration, whisper_model=whisper_model,
                whisper_mode=whisper_mode, progress_node_id=progress_node_id)
        if reference_quality_gate == 'fail_on_strong_reference_leakage' and diagnostic_mode != 'segmented':
            raise ValueError('A strict Reference Voice lexical gate requires diagnostic_mode=segmented')
        selected=validate_reference_models(
            ckpt_name,audio_encoder_name,mert_model,tokenizer_name,
            nar_lora_name,whisper_model,reference_lyrics,whisper_mode=whisper_mode)
        print('[YuE2 Reference Voice][SELECTED MODELS] '+json.dumps(selected,ensure_ascii=False))
        report_progress(progress_node_id,'models_validated')
        try:
            from comfy_execution.graph_utils import GraphBuilder
        except ImportError as exc:
            raise RuntimeError('Update ComfyUI to a version supporting GraphBuilder and native YuE2 nodes.') from exc

        from .reference_strategy import effective_style, CONDITIONING_MODES
        if conditioning_mode not in CONDITIONING_MODES:
            raise ValueError('Unsupported Reference Voice conditioning_mode: '+str(conditioning_mode))
        if legacy_prefix_behavior:
            # Optional A/B comparison with the original full-prefix behavior. Keep the same seed,
            # sample, lyrics and models; no implicit Style addition or clip trim.
            effective=str(style or '')
            style_strategy={'policy':'legacy_reference_prefix','target_language':'unchanged',
                            'user_style_provided':bool(effective.strip()),
                            'effective_style_nonempty':bool(effective.strip()),
                            'lyrics_injected_into_style':False}
            reference_clip_policy='full'
            conditioning_mode='full'
            budget_policy='user_maximum'
        else:
            effective,style_strategy=effective_style(style,lyrics,target_language)
            style_strategy['policy']='native_cover_controlled'
            budget_policy='score_bounded'
        graph=GraphBuilder()
        reference_clip=graph.node('YuE2NativeSelectReferenceClip',reference_audio=reference_audio,
            reference_lyrics=reference_lyrics,clip_policy=reference_clip_policy,
            clip_seconds=reference_clip_seconds)
        model=graph.node('CheckpointLoaderSimple',ckpt_name=ckpt_name)
        encoder=graph.node('AudioEncoderLoader',audio_encoder_name=audio_encoder_name)
        source_score=graph.node('SheetSage2AudioToABC',audio_encoder=encoder.out(0),
                                 audio=source_audio,mode='melody')
        reference_score=graph.node('SheetSage2AudioToABC',audio_encoder=encoder.out(0),
                                    audio=reference_clip.out(0),mode='melody')
        ref_words=graph.node('YuE2NativeReferenceLyrics',reference_audio=reference_clip.out(0),
                             reference_lyrics=reference_lyrics,whisper_model=whisper_model,whisper_mode=whisper_mode,
                             progress_node_id=progress_node_id)
        # Use the same effective arrangement on target-score fallback and
        # final CLIP conditioning. An empty user Style previously disabled
        # the native YuE2 GenerateABC recovery even with CLIP available.
        target_vocal=graph.node('YuE2AutoVocalPRO',abc=source_score.out(0),
                    lyrics=lyrics,style=effective,clip=model.out(1),seed=seed,
                    strict=False,reference_composite=True,progress_node_id=progress_node_id,
                    max_duration=max_duration,source_audio=source_audio)
        ref_vocal=graph.node('YuE2AutoVocalPRO',abc=reference_score.out(0),lyrics=ref_words.out(0),
                    section_policy='all_sections',instrumental_handling='replace_with_rests',
                    existing_vocal='preserve',instrumental_tags='',vocal_tags='',strict=True,
                    reference_sample_only=True,
                    )
        plan=graph.node('YuE2NativeReferencePlan',target_abc=target_vocal.out(0),
                    reference_abc=ref_vocal.out(0),lyrics=target_vocal.out(1),
                    reference_lyrics=ref_words.out(0),reference_audio=reference_clip.out(0),
                    first_note_seconds=first_note_seconds,progress_node_id=progress_node_id,
                    target_plan_report=target_vocal.out(2))
        plan_trace=graph.node('YuE2NativeReferenceTracePlan',
                    plan_report=plan.out(4),composite_lyrics=plan.out(1),
                    reference_lyrics=ref_words.out(0),target_lyrics=lyrics)
        tokens=graph.node('YuE2NativeReferenceTokens',aligned_reference_audio=plan.out(2),
                          mert_model=mert_model,tokenizer_name=tokenizer_name,
                          progress_node_id=progress_node_id)
        cond=graph.node('YuE2NativeReferenceConditioning',clip=model.out(1),
                    style=effective,lyrics=plan.out(1),abc=plan.out(0),semantic_prefix=tokens.out(0),
                    # Mode is explicit; legacy workflows default to full for reproducibility.
                    seed=seed,mode=conditioning_mode,max_duration=max_duration,
                    budget_policy=budget_policy,
                    prefix_seconds=plan.out(3),plan_report=plan_trace.out(0),progress_node_id=progress_node_id)
        negative=graph.node('ConditioningZeroOut',conditioning=cond.out(0))
        latent=graph.node('EmptyYuE2LatentAudio',seconds=cond.out(1),batch_size=1)
        acoustic_model=graph.node('LoraLoaderModelOnly',model=model.out(0),lora_name=nar_lora_name,
                                  strength_model=1.0)
        sampling_input=graph.node('YuE2NativeProgressLatent',latent=latent.out(0),
                    stage='nar_sampling_started',progress_node_id=progress_node_id)
        sampler=graph.node('KSampler',model=acoustic_model.out(0),positive=cond.out(0),
                           negative=negative.out(0),latent_image=sampling_input.out(0),seed=seed,
                           steps=64,cfg=1.0,sampler_name='dpm_2',scheduler='sgm_uniform',denoise=1.0)
        sampling_complete=graph.node('YuE2NativeProgressLatent',latent=sampler.out(0),
                    stage='nar_sampling_completed',progress_node_id=progress_node_id)
        decoded=graph.node('VAEDecodeAudio',samples=sampling_complete.out(0),vae=model.out(2))
        cropped=graph.node('YuE2NativeTrimReference',audio=decoded.out(0),prefix_seconds=plan.out(3),min_output_seconds=cond.out(3),
                    progress_node_id=progress_node_id)
        diag=graph.node('YuE2NativeExecutionReport',audio=cropped.out(0),
                        plan_report=plan_trace.out(0),lyrics_report=ref_words.out(1),
                        tokenizer_report=tokens.out(1),ar_report=cond.out(2),
                        ckpt_name=ckpt_name,audio_encoder_name=audio_encoder_name,
                        mert_model=mert_model,tokenizer_name=tokenizer_name,
                        nar_lora_name=nar_lora_name,whisper_model=whisper_model,whisper_mode=whisper_mode,
                        seed=seed,target_lyrics=lyrics,reference_text=ref_words.out(0),
                        source_audio=source_audio,reference_audio=reference_audio,
                        source_contamination_audit=source_contamination_audit,
                        diagnostic_mode=diagnostic_mode,diagnostics_include_text=diagnostics_include_text,
                        reference_clip_report=reference_clip.out(1),
                        reference_style_report=json.dumps(style_strategy,ensure_ascii=False),
                        reference_quality_gate=reference_quality_gate,
                        model_paths=json.dumps(selected,ensure_ascii=False),
                        progress_node_id=progress_node_id)
        return {'result':(diag.out(0),diag.out(1)), 'expand':graph.finalize()}


_REPORT_UNCONNECTED = object()  # Distinguish legacy workflows from an unresolved lazy link.


class YuE2CoverModeSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'mode':(['standard','reference_voice'],{'default':'standard'}),
                            'standard_audio':('AUDIO',{'lazy':True}),
                            'reference_audio':('AUDIO',{'lazy':True})},
                'optional':{'reference_report':('STRING',{'lazy':True})}}
    RETURN_TYPES=('AUDIO','STRING')
    RETURN_NAMES=('AUDIO','selected_mode_report')
    FUNCTION='choose'; CATEGORY='audio/YuE2'
    @classmethod
    def check_lazy_status(cls,mode,standard_audio=None,reference_audio=None,reference_report=_REPORT_UNCONNECTED):
        if mode=='standard':
            return ['standard_audio'] if standard_audio is None else []
        if mode=='reference_voice':
            missing=[]
            if reference_audio is None: missing.append('reference_audio')
            if reference_report is None: missing.append('reference_report')
            return missing
        raise ValueError(f'Unsupported cover mode: {mode}')
    def choose(self,mode,standard_audio=None,reference_audio=None,reference_report=_REPORT_UNCONNECTED):
        value=standard_audio if mode=='standard' else reference_audio
        if value is None:
            raise RuntimeError('Selected audio branch is not connected')
        if mode=='standard':
            summary=json.dumps({'selected_mode':'standard','reference_voice_executed':False,
                                'note':'Reference Voice branch was not evaluated in standard mode.'},
                               ensure_ascii=False,indent=2)
        else:
            if reference_report is _REPORT_UNCONNECTED:
                # Existing/legacy workflows had only the AUDIO output. Keep working
                # without falsely claiming a completed execution audit.
                summary=json.dumps({'selected_mode':'reference_voice','audit_available':False,
                                    'note':'Legacy workflow did not connect the optional execution report'},
                                   ensure_ascii=False,indent=2)
            elif not reference_report:
                raise RuntimeError('Reference Voice report is connected but the execution trace is empty')
            else:
                summary=reference_report
        return value,summary


NODE_CLASS_MAPPINGS={'YuE2ReferenceVoiceCover':YuE2ReferenceVoiceCover,
                     'YuE2CoverModeSelector':YuE2CoverModeSelector}
NODE_DISPLAY_NAME_MAPPINGS={'YuE2ReferenceVoiceCover':'YuE2 Reference Voice Cover (native prefix)',
                            'YuE2CoverModeSelector':'YuE2 Cover — Standard / Reference Voice'}
