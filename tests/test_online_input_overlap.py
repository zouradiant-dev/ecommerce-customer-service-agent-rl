import json
import unittest
from ecommerce_customer_service.serving.context_builder import ContextBuilder,HFTokenCounter
from ecommerce_customer_service.serving.policy import get_policy
from ecommerce_customer_service.serving.repair_sft import HANDOFF_REQUESTS,episode
from ecommerce_customer_service.serving.split_integrity import audit


class CharacterTokenizer:
    def apply_chat_template(self,messages,tools=None,add_generation_prompt=False,**kwargs):
        return list(json.dumps([messages,tools,add_generation_prompt],ensure_ascii=False,sort_keys=True).encode())


class OverlapTests(unittest.TestCase):
    def row(self,split,text,target='回答'):
        policy=get_policy('binding_compact_v3'); tokenizer=CharacterTokenizer()
        context=ContextBuilder(counter=HFTokenCounter(tokenizer))
        built=context.build(policy.prompt,[{'role':'user','content':text}],policy.tools,
                            {'conversation_status':'active','operations':[],'order_observations':[]})
        return {'id':split+'_different_id','split':split,'tools':policy.tools,
                'messages':built.messages+[{'role':'assistant','content':target}]}

    def test_actual_inputs_collide_despite_different_ids_and_targets(self):
        result=audit({'train':[self.row('train','转人工','目标一')],
                      'dev':[self.row('dev','转人工','目标二')]},CharacterTokenizer())
        self.assertFalse(result['passed']);self.assertEqual(1,len(result['cross_split_collisions']))

    def test_sealed_first_turn_collision_is_detected_without_reference_execution(self):
        case=episode('test','handoff',0,7)
        row=self.row('train',case['steps'][0]['text'])
        case.pop('smoke_responses');case.pop('expected')
        result=audit({'train':[row],'dev':[]},CharacterTokenizer(),sealed_cases=[case])
        self.assertFalse(result['passed']);self.assertFalse(result['test_reference_executed'])

    def test_new_handoff_expressions_are_disjoint_and_pass_rendered_input_check(self):
        self.assertFalse(set(HANDOFF_REQUESTS['train']) & set(HANDOFF_REQUESTS['dev']))
        result=audit({'train':[self.row('train',HANDOFF_REQUESTS['train'][0])],
                      'dev':[self.row('dev',HANDOFF_REQUESTS['dev'][0])]},CharacterTokenizer(),
                     sealed_cases=[episode('test','handoff',0,7)])
        self.assertTrue(result['passed'])

    def test_split_labels_cannot_be_misrepresented(self):
        with self.assertRaisesRegex(ValueError,'invalid_supervision_split'):
            audit({'dev':[self.row('train','查询')]},CharacterTokenizer())
        with self.assertRaisesRegex(ValueError,'train_dev_only'):
            audit({'test':[]},CharacterTokenizer())


if __name__=='__main__':unittest.main()
