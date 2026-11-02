import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'custom_nodes'))
from yue2_auto_vocal_pro.engine import convert_abc, measure_rests

SYNTHETIC = '''X:1
T:Synthetic Example
M:4/4
L:1/16
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% intro
V: Vocal
Z2|
V: Ins
C16|G16|
% verse
V: Vocal
Z4|
V: Ins
C4D4E4F4|G8E8|D8C8|C16|
V: Vocal
Z4|
V: Ins
E4F4G4A4|G8E8|D8C8|C16|
% interlude
V: Vocal
Z2|
V: Ins
G16|C16|
% verse
V: Vocal
Z4|
V: Ins
C4C4G4G4|A8G8|F8E8|D16|
% outro
V: Vocal
Z2|
V: Ins
G16|C16|
'''
WORDS = '[Verse]\nWe sing these original words\nAnother clear line of the song\n\n[Chorus]\nSing this refrain again\nAnd end with words'

class EngineTest(unittest.TestCase):
    def test_promote_exact_native_format(self):
        score, text, report = convert_abc(SYNTHETIC, WORDS)
        self.assertEqual(WORDS, text)
        self.assertIn('V: Vocal\nC4D4E4F4|G8E8|D8C8|C16|\nV: Ins\nZ4|', score)
        self.assertIn('% intro\nV: Vocal\nZ2|', score)
        self.assertIn('% interlude\nV: Vocal\nZ2|', score)
        self.assertIn('% outro\nV: Vocal\nZ2|', score)
        self.assertIn('converted_groups=3', report)
        self.assertIn('converted_bars=12', report)
        self.assertIn('expanded_placeholders=3', report)
    def test_idempotent(self):
        score, _, _ = convert_abc(SYNTHETIC, WORDS)
        score2, _, report = convert_abc(score, WORDS, existing_vocal='preserve')
        self.assertEqual(score,score2)
        self.assertIn('existing_vocal_groups=3',report)
    def test_keep_original(self):
        score, _, _=convert_abc(SYNTHETIC, WORDS, instrumental_handling='keep_original')
        self.assertIn('V: Vocal\nC4D4E4F4|G8E8|D8C8|C16|\nV: Ins\nC4D4E4F4|G8E8|D8C8|C16|',score)
    def test_natural_rest(self):
        self.assertEqual(measure_rests('4/4','1/16',4),'Z4|')
        self.assertEqual(measure_rests('3/4','1/16',2),'Z2|')
    def test_chords(self):
        src='X:1\nM:4/4\nL:1/16\nK:C\n% verse\nV: Vocal\n"C"z12"G"z4|\nV: Ins\nC4D4E4F4|\n'
        score, _, _=convert_abc(src,WORDS)
        self.assertIn('"C"C4D4E4"G"F4|',score)
    def test_empty_lyrics_rejected(self):
        with self.assertRaisesRegex(ValueError,'Lyrics are empty'):
            convert_abc(SYNTHETIC,'',strict=True)
    def test_empty_can_warn_when_non_strict(self):
        _, _, report = convert_abc(SYNTHETIC,'',strict=False)
        self.assertIn('No lyric words found',report)
    def test_instrumental_override(self):
        score,_,report=convert_abc(SYNTHETIC,WORDS,instrumental_tags='verse',strict=False)
        self.assertIn('converted_groups=0',report)
        self.assertIn('% verse\nV: Vocal\nZ4|',score)
    def test_instrumental_section_forced_vocal(self):
        score,_,report=convert_abc(SYNTHETIC,WORDS,vocal_tags='intro')
        self.assertIn('% intro\nV: Vocal\nC16|G16|',score)
        self.assertIn('converted_groups=4',report)
    def test_synthetic_no_copyright_fixture(self):
        self.assertNotIn('Zelda',SYNTHETIC)

if __name__=='__main__': unittest.main()
