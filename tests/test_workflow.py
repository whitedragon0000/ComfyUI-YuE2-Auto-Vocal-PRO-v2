import json
import unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
FILE=BASE/'YuE2_Cover_AutoVocal_PRO_V2.json'

class WorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(FILE.read_text(encoding='utf8'))
        cls.sub=cls.data['definitions']['subgraphs'][0]
        cls.outer={n['id']:n for n in cls.data['nodes']}
        cls.inner={n['id']:n for n in cls.sub['nodes']}
        cls.links={l['id']:l for l in cls.sub['links']}
    def test_name_and_ids_unchanged(self):
        self.assertEqual(self.sub['name'],'Music Cover (YuE2) — Auto Vocal PRO v2')
        self.assertEqual(self.sub['id'],'59dddde1-2d83-4b6e-9c71-1e5cb2d823d6')
        self.assertEqual(len(self.outer),6)
        self.assertEqual(len(self.inner),14)
        self.assertEqual(self.inner[47]['type'],'YuE2AutoVocalPRO')
    def test_outer_unchanged_ports_and_links(self):
        self.assertEqual(len(self.outer[33]['inputs']),14)
        self.assertEqual(len(self.sub['inputs']),14)
        self.assertEqual([a[0] for a in self.data['links']],[97,117,63,118])
        self.assertEqual(self.outer[33]['inputs'][1]['link'],118)
        self.assertEqual(self.outer[33]['inputs'][3]['link'],117)
        self.assertEqual(self.outer[33]['inputs'][5]['link'],97)
    def test_empty_fields(self):
        self.assertEqual(self.outer[33]['widgets_values_named']['style'],'')
        self.assertEqual(self.outer[33]['widgets_values'][0],'')
        self.assertEqual(self.outer[50]['widgets_values_named']['value'],'')
        self.assertEqual(self.outer[50]['widgets_values'][0],'')
    def test_original_defaults_preserved(self):
        self.assertEqual(self.outer[33]['widgets_values_named']['max_duration'],190)
        self.assertEqual(self.outer[33]['widgets_values_named']['planning_mode'],'melody')
        self.assertEqual(self.outer[33]['widgets_values_named']['section_policy'],'all_of_rest')
        self.assertEqual(self.outer[51]['widgets_values_named']['control_after_generate'],'randomize')
    def test_node_input_added_only_inside(self):
        self.assertEqual(len(self.inner[47]['inputs']),9)
        self.assertEqual(self.inner[47]['inputs'][-1]['name'],'lyrics_alignment')
        self.assertEqual(self.inner[47]['inputs'][-1]['link'],None)
        self.assertEqual(self.inner[47]['widgets_values_named']['lyrics_alignment'],'match_sections')
    def test_all_original_subgraph_links_resolve(self):
        self.assertEqual(len(self.links),len(self.sub['links']))
        for lid,edge in self.links.items():
            self.assertIn(edge['origin_id'],self.inner|{-10:{},-20:{}})
            self.assertIn(edge['target_id'],self.inner|{-10:{},-20:{}})
            if edge['target_id'] not in (-10,-20):
                self.assertEqual(self.inner[edge['target_id']]['inputs'][edge['target_slot']]['link'],lid)
            if edge['origin_id'] not in (-10,-20):
                self.assertIn(lid,self.inner[edge['origin_id']]['outputs'][edge['origin_slot']]['links'] or [])
    def test_audio_and_text_paths(self):
        self.assertEqual((self.links[96]['origin_id'],self.links[96]['target_id']),(-10,47))
        self.assertEqual((self.links[114]['origin_id'],self.links[114]['target_id']),(47,25))
        self.assertEqual((self.links[104]['origin_id'],self.links[104]['target_id']),(47,25))
        self.assertEqual((self.links[115]['origin_id'],self.links[115]['target_id']),(47,49))
        self.assertEqual((self.links[106]['origin_id'],self.links[106]['target_id']),(47,48))
        self.assertEqual((self.links[68]['origin_id'],self.links[69]['origin_id']),(34,34))
    def test_no_new_models_or_audio_in_repo(self):
        for path in BASE.rglob('*'):
            self.assertNotIn(path.suffix.lower(),{'.mp3','.mp4','.flac','.safetensors','.ckpt'})

if __name__=='__main__':unittest.main()
