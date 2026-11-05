"""ComfyUI custom node to auto-route SheetSage2 melody into YuE2 Vocal."""
from .engine import convert_abc
from .alignment import align_lyrics
from .lyric_contract import automatic_lyrics, measure, with_terminal_end, with_instrumental_tail

class YuE2AutoVocalPRO:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'abc': ('STRING', {'forceInput': True}),
            'lyrics': ('STRING', {'forceInput': True}),
            'section_policy': (['all_of_rest','verses_and_choruses','all_sections'], {'default':'all_of_rest'}),
            'instrumental_handling': (['replace_with_rests','keep_original'], {'default':'replace_with_rests'}),
            'existing_vocal': (['preserve','overwrite'], {'default':'preserve'}),
            'instrumental_tags': ('STRING',{'multiline':True,'default':'','tooltip':'Extra section tags to keep instrumental, comma/newline separated'}),
            'vocal_tags': ('STRING', {'multiline': True,'default':'','tooltip':'Extra section tags treated as vocal, comma/newline separated'}),
            'strict': ('BOOLEAN', {'default':True}),
            'vocal_melody_mode': (['Adaptive', 'Music adapts to Lyrics', 'Lyrics adapt to Music'], {'default':'Adaptive'}),
            # Lyrics are always planned automatically; there is no user policy switch.
        },
        'optional': {
            'clip': ('CLIP',),
            'style': ('STRING', {'forceInput': True}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'progress_node_id': ('STRING', {'default': '', 'forceInput': True}),
            'reference_composite': ('BOOLEAN', {'forceInput': True}),
            'max_duration': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 900.0, 'tooltip': 'Maximum, not exact target length.'}),
            'source_audio': ('AUDIO',),
            'reference_sample_only': ('BOOLEAN', {'default':False}),
            'forensic_replay': ('BOOLEAN', {'default':False, 'tooltip':'ONLY for bug reports: include ABC and Lyrics in clear text in diagnostic ZIP. No audio.'}),
        }}
    RETURN_TYPES=('STRING','STRING','STRING')
    RETURN_NAMES=('abc','lyrics','report')
    FUNCTION='route'
    CATEGORY='audio/YuE2'
    DESCRIPTION='Preserves a valid native source score and every target lyric word. Regenerates ABC only if the source score cannot be used, never from syllable-count heuristics.'
    def route(self, abc, lyrics, section_policy='all_of_rest',
              instrumental_handling='replace_with_rests', existing_vocal='preserve',
              instrumental_tags='', vocal_tags='', strict=True, vocal_melody_mode='Adaptive', lyrics_alignment=None,
              clip=None, style='', seed=0, progress_node_id='', reference_composite=False, max_duration=0.0, source_audio=None, reference_sample_only=False, forensic_replay=False):
        """Deterministic score/lyrics planning with bounded official-ABC fallback.

        Optional CLIP/Style enables the stock YuE2GenerateABC only for unfit or
        unparseable source scores. No manual melody/lyrics policy UI remains.
        """
        from .reference_progress import report_progress
        from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc
        import json
        from .instrumental_mode import instrumental_requested, instrumentalize_score, END
        report_progress(progress_node_id, 'score_selection_started')
        if instrumental_requested(lyrics) and not reference_sample_only:
            instrumental_abc, evidence = instrumentalize_score(abc, max_duration=max_duration)
            evidence['score_source'] = 'SheetSage2'
            evidence['reference_voice_bypassed'] = bool(reference_composite)
            report_progress(progress_node_id, 'score_selection_completed')
            return instrumental_abc, END, json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        if instrumental_requested(lyrics):
            raise ValueError('The Reference Voice sample must contain sung words; only the target may be instrumental')
        if reference_sample_only:
            # Preserve the GOOD reference-voice implementation's score conversion.
            # The reference clip is a timbre/phonetic prefix, NOT the new song.
            # Never run its notes or lyrics through the adaptive target planner.
            converted_ref = convert_abc(
                abc, lyrics, section_policy=section_policy,
                instrumental_handling=instrumental_handling,
                existing_vocal=existing_vocal,
                instrumental_tags=instrumental_tags, vocal_tags=vocal_tags,
                strict=True)
            ref_abc, ref_lyrics, section_report = align_lyrics(
                converted_ref[0], converted_ref[1], mode='report_only')
            parse_abc(ref_abc)
            from .lyric_contract import _tokens
            assert _tokens(ref_lyrics)[0] == _tokens(lyrics)[0], 'Reference sample words changed'
            ref_report = {
                'decision': 'preserve_known_good_reference_sample',
                'reference_sample_only': True,
                'vocal_melody_mode': 'Lyrics adapt to Music',
                'vocal_melody_alignment': {'score_changed': False, 'target_planner_executed': False},
                'score_conversion_report': converted_ref[2],
                'section_report': section_report}
            report_progress(progress_node_id, 'score_selection_completed')
            return ref_abc, ref_lyrics, json.dumps(ref_report, ensure_ascii=False, sort_keys=True)
        # ONE production score contract for Standard and Reference Voice.
        # An ABC can be musical and valid even when there are fewer/new Lyrics
        # or fewer syllables than notes. These are diagnostic, not hard errors.
        from .cover_contract import prepare_cover
        from .lyric_contract import _tokens
        from .vendor.yue2_cover_main.yue2_cover.abc_tools import parse_abc
        errors = []
        try:
            converted = convert_abc(abc, lyrics, section_policy=section_policy,
                instrumental_handling=instrumental_handling,
                existing_vocal=existing_vocal, instrumental_tags=instrumental_tags,
                vocal_tags=vocal_tags, strict=False)
            prepared = prepare_cover(converted[0], lyrics, max_duration=max_duration,
                                     origin='SheetSage2')
        except ValueError as exc:
            errors.append(str(exc))
            # Regeneration is reserved for a *genuinely unusable* ABC,
            # not for a syllable heuristic, short Lyrics, or a user tag.
            # If we have a valid musical source but cannot honour an actual
            # maximum, produce an explicit actionable error rather than
            # fabricate an unrelated song with YuE2GenerateABC.
            valid_source = False
            try:
                parsed = parse_abc(converted[0] if 'converted' in locals() else abc)
                valid_source = bool(parsed.voices['Vocal'].notes)
            except ValueError:
                pass
            if valid_source or clip is None:
                raise ValueError('Source-score cover contract failed: ' + str(exc)) from exc
            from comfy_execution.graph_utils import GraphBuilder
            graph = GraphBuilder()
            generated = graph.node('YuE2GenerateABC', clip=clip,
                style=style or 'faithful source melody, natural vocal phrasing',
                lyrics=with_terminal_end(lyrics), seed=int(seed), mode='melody',
                max_abc_tokens=8192)
            checked = graph.node('YuE2NativeValidateGeneratedScore',
                generated_abc=generated.out(0), source_abc=abc,
                lyrics=lyrics, source_report=str(exc),
                reason='source_abc_unusable', progress_node_id=progress_node_id,
                reference_composite=reference_composite, max_duration=max_duration,
                vocal_melody_mode=vocal_melody_mode,
                section_policy=section_policy,
                instrumental_handling=instrumental_handling,
                existing_vocal=existing_vocal,
                instrumental_tags=instrumental_tags,
                vocal_tags=vocal_tags)
            print('[YuE2 Auto Vocal PRO][AUTO SCORE] Native regeneration requested '
                  'because the source ABC is unusable: ' + str(exc))
            return {'result': (checked.out(0), checked.out(1), checked.out(2)),
                    'expand': graph.finalize()}
        from .lyric_contract import measure
        advisory = measure(prepared.abc, lyrics,
            max_duration=max_duration if max_duration > 0 else None)
        audit = dict(prepared.report)
        audit['decision'] = 'source_musical_excerpt' if prepared.report['score_durations']['excerpt'] else 'preserve_source_melody'
        audit['vocal_melody_alignment'] = {'word_order_unchanged': True,
            'action': audit['score_durations']['action'], 'mode': vocal_melody_mode,
            'duration_fit': {**audit['score_durations'],
                'matched_section': (audit['score_durations'].get('source_trace',{}).get('target_section_labels') or [None])[0]},
            'score_changed': prepared.abc != converted[0],
            'acoustic_fidelity_guaranteed': False}
        audit['warning'] = '; '.join(audit['section_contract']['warnings']) or None
        audit['fitness'] = advisory.__dict__
        audit['conversion_report'] = converted[2]
        audit['vocal_melody_mode'] = vocal_melody_mode
        audit['reference_composite'] = bool(reference_composite)
        audit['score_recovery'] = (prepared.report['score_durations'].get('source_trace')
                                   if prepared.report['score_durations']['excerpt'] else None)
        audit['word_count'] = len(_tokens(lyrics)[0])
        audit['score_changed'] = prepared.abc != converted[0]
        audit['musical_note_event_inventory_preserved'] = True
        audit['arrangement_change_type'] = ('duration_excerpt' if audit['score_durations']['excerpt']
             else 'instrumental_revoicing' if audit['instrumental_arrangement']['groups_revoiced'] else 'none')
        report_progress(progress_node_id, 'score_selection_completed')
        print('[YuE2 Auto Vocal PRO][SCORE] ' + json.dumps({
            'decision': audit['decision'],
            'score_source': audit['score_source'],
            'score_seconds': audit['score_durations']['selected_seconds'],
            'max_duration': max_duration,
            'lyrics_words': audit['word_count'],
            'empty_vocal_sections': audit['section_contract']['empty_vocal_sections_after_arrangement'],
            'instrumental_groups_revoiced': audit['instrumental_arrangement']['groups_revoiced'],
            'warnings': audit['section_contract']['warnings'][:3]}, ensure_ascii=False))
        return prepared.abc, with_terminal_end(prepared.lyrics), json.dumps(audit, ensure_ascii=False)

from .vocal_melody_engine import YuE2VocalMelodyAlignmentEngine
NODE_CLASS_MAPPINGS={'YuE2AutoVocalPRO':YuE2AutoVocalPRO,
                     'YuE2VocalMelodyAlignmentEngine':YuE2VocalMelodyAlignmentEngine}
NODE_DISPLAY_NAME_MAPPINGS={'YuE2AutoVocalPRO':'YuE2 Auto Vocal PRO (ABC + Lyrics)',
    'YuE2VocalMelodyAlignmentEngine':'YuE2 Vocal-Melody Alignment Engine'}

# Optional isolated reference-voice generator; importing it loads no GPU models.
from .voice_reference import NODE_CLASS_MAPPINGS as _REF_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as _REF_NAMES
NODE_CLASS_MAPPINGS.update(_REF_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_REF_NAMES)

# Optional download/check node; registered without importing model libraries.
from .model_store import NODE_CLASS_MAPPINGS as _MODEL_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as _MODEL_NAMES
NODE_CLASS_MAPPINGS.update(_MODEL_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_MODEL_NAMES)

from .native_nodes import NODE_CLASS_MAPPINGS as _NATIVE_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as _NATIVE_NAMES
NODE_CLASS_MAPPINGS.update(_NATIVE_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_NATIVE_NAMES)

# Frontend does not draw progress overlays. Native ComfyUI progress is authoritative.
WEB_DIRECTORY = "./web"

from .instrumental_graph import NODE_CLASS_MAPPINGS as _INST_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as _INST_NAMES
NODE_CLASS_MAPPINGS.update(_INST_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_INST_NAMES)

from .instrumental_mode import NODE_CLASS_MAPPINGS as _STYLE_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as _STYLE_NAMES
NODE_CLASS_MAPPINGS.update(_STYLE_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_STYLE_NAMES)

from .vocal_fidelity import NODE_CLASS_MAPPINGS as _AUDIT_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as _AUDIT_NAMES
NODE_CLASS_MAPPINGS.update(_AUDIT_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_AUDIT_NAMES)

from .duration_contract import NODE_CLASS_MAPPINGS as _DURATION_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as _DURATION_NAMES
NODE_CLASS_MAPPINGS.update(_DURATION_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_DURATION_NAMES)

# Forensic recorder decorates only node methods: no change to graph logic or outputs.
from .forensic_recorder import install as _install_forensic_recorder
_install_forensic_recorder(NODE_CLASS_MAPPINGS)
