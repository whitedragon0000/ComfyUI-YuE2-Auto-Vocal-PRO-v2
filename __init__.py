"""ComfyUI custom node to auto-route SheetSage2 melody into YuE2 Vocal."""
from .engine import convert_abc
from .alignment import align_lyrics

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
            'lyrics_alignment': (['match_sections','report_only','repeat_existing'], {'default':'match_sections','tooltip':'match_sections aligns only ABC section labels; repeat_existing ALSO repeats your exact lyric lines on explicit selection; report_only changes neither.'}),
        }}
    RETURN_TYPES=('STRING','STRING','STRING')
    RETURN_NAMES=('abc','lyrics','report')
    FUNCTION='route'
    CATEGORY='audio/YuE2'
    DESCRIPTION='Routes SheetSage2 melody to Vocal; diagnoses lyrical coverage and optionally aligns ABC section labels or repeats only existing lyric lines.'
    def route(self, abc, lyrics, section_policy='all_of_rest',instrumental_handling='replace_with_rests',existing_vocal='preserve',instrumental_tags='',vocal_tags='',strict=True,lyrics_alignment='match_sections'):
        output=convert_abc(abc,lyrics,section_policy=section_policy,
            instrumental_handling=instrumental_handling,existing_vocal=existing_vocal,
            instrumental_tags=instrumental_tags,vocal_tags=vocal_tags,strict=strict)
        aligned_abc, aligned_lyrics, alignment_report = align_lyrics(output[0], output[1], mode=lyrics_alignment)
        report = output[2] + '\n' + alignment_report
        print('[YuE2 Auto Vocal PRO]\n'+report)
        return aligned_abc, aligned_lyrics, report

NODE_CLASS_MAPPINGS={'YuE2AutoVocalPRO':YuE2AutoVocalPRO}
NODE_DISPLAY_NAME_MAPPINGS={'YuE2AutoVocalPRO':'YuE2 Auto Vocal PRO (ABC + Lyrics)'}
