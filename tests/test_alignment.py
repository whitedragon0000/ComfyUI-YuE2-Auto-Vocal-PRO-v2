import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'custom_nodes'))
from yue2_auto_vocal_pro.alignment import align_lyrics, score_runs, lyric_parts
from yue2_auto_vocal_pro.engine import convert_abc
from yue2_auto_vocal_pro import YuE2AutoVocalPRO
from test_engine import SYNTHETIC, WORDS

class LyricsAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routed,_,_=convert_abc(SYNTHETIC,WORDS)
    def test_two_distinct_vocal_runs(self):
        runs=score_runs(self.routed)
        self.assertEqual([(r.label,r.bars) for r in runs],[('verse',8),('verse',4)])
        self.assertTrue(all(r.onsets > 0 for r in runs))
    def test_match_sections_no_lyric_rewrites(self):
        scored, lyr, report=align_lyrics(self.routed,WORDS,mode='match_sections')
        self.assertEqual(lyr,WORDS)
        self.assertIn('% chorus\nV: Vocal\nC4C4G4G4|',scored)
        self.assertIn('% interlude\nV: Vocal\nZ2|',scored)
        self.assertIn('Section label: %verse -> %chorus',report)
    def test_report_only_byte_identical(self):
        s,l,r=align_lyrics(self.routed,WORDS,mode='report_only')
        self.assertEqual(s,self.routed)
        self.assertEqual(l,WORDS)
        self.assertIn('vocal_sections=2',r)
    def test_mismatch_does_not_guess(self):
        lyr='[Verse]\nWords once\n[Verse]\nWords twice\n[Chorus]\nAnd more words'
        s,l,r=align_lyrics(self.routed,lyr,mode='repeat_existing')
        self.assertEqual(s,self.routed)
        self.assertEqual(l,lyr)
        self.assertIn('WARNING: 2 ABC vocal sections vs 3 lyric sections',r)
    def test_repeat_existing_explicit_only(self):
        s,l,r=align_lyrics(self.routed,WORDS,mode='repeat_existing')
        self.assertEqual(s.count('% chorus'),1)
        # Verse has 8 bars for 2 lines -> repeat 2x. Chorus 4 bars -> no repeat.
        self.assertEqual(l.count('We sing these original words'),2)
        self.assertEqual(l.count('And end with words'),1)
        self.assertIn('explicitly repeated existing lines',r)
    def test_default_does_not_repeat(self):
        s,l,r=align_lyrics(self.routed,WORDS)
        self.assertEqual(l,WORDS)
    def test_unlabelled_lyrics_single_section(self):
        s,l,r=align_lyrics(self.routed,'One line no heading')
        self.assertEqual(l,'One line no heading')
        self.assertIn('WARNING: 2 ABC vocal sections vs 1 lyric sections',r)
    def test_custom_node_signature(self):
        signature=YuE2AutoVocalPRO.INPUT_TYPES()['required']
        self.assertEqual(list(signature)[-1],'lyrics_alignment')
        self.assertEqual(signature['lyrics_alignment'][1]['default'],'match_sections')
        self.assertEqual(YuE2AutoVocalPRO.RETURN_NAMES,('abc','lyrics','report'))
    def test_full_node_real_call(self):
        score, lyrics, report=YuE2AutoVocalPRO().route(SYNTHETIC,WORDS,lyrics_alignment='match_sections')
        self.assertEqual(lyrics,WORDS)
        self.assertIn('% chorus',score)
        self.assertIn('converted_bars=12',report)
        self.assertIn('Lyrics alignment: mode=match_sections',report)
    def test_preserve_arbitrary_suno_style_tags(self):
        lyric='[Verse 1 - expressive lead]\nhello again\n[Final Chorus - big backing vocals]\nfinally home'
        a,b,c=align_lyrics(self.routed,lyric)
        self.assertEqual(b,lyric)
        self.assertIn('% chorus',a)
    def test_diagnostics_for_sparser_lyrics(self):
        a,b,c=align_lyrics(self.routed,'[Verse]\nline\n[Chorus]\nline')
        self.assertIn('WARNING section 1: sparse lyrics',c)
    def test_invalid_mode(self):
        with self.assertRaises(ValueError):align_lyrics(self.routed,WORDS,mode='invent_new_words')

if __name__=='__main__':unittest.main()
