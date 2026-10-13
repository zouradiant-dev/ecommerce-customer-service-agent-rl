from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from ecommerce_customer_service.serving.repair_sft import KINDS, PRODUCTS, episode, execute_reference
from ecommerce_customer_service.training import chat_template_ids, render_with_assistant_mask, validate_training_row


class RepairDataTests(unittest.TestCase):
    def test_disjoint_splits_and_candidate_labels_remain_correct_with_many_orders(self):
        catalogs = [set(product for product, _ in PRODUCTS[split]) for split in PRODUCTS]
        for index, catalog in enumerate(catalogs):
            for other in catalogs[index+1:]: self.assertFalse(catalog & other)
        seen = set()
        for split in PRODUCTS:
            identifiers = set()
            amount_directions = set()
            for kind in KINDS:
                for variant in range(8):
                    case = episode(split, kind, variant, 7)
                    self.assertEqual(case, episode(split, kind, variant, 7))
                    identifiers.update(order['id'] for order in case['initial_orders'])
                    owned = [order for order in case['initial_orders'] if order['customer_id'] == 'customer_alice']
                    count = sum(order['product'] == owned[0]['product'] for order in owned)
                    self.assertEqual(2 if kind == 'true_ambiguity' else 1, count)
                    if kind == 'change_target':
                        self.assertEqual(1, sum(order['product'] == owned[1]['product'] for order in owned))
                    if kind == 'true_ambiguity':
                        amount_directions.add(owned[0]['paid_cents'] < owned[1]['paid_cents'])
            self.assertEqual({False,True},amount_directions)
            self.assertFalse(seen & identifiers)
            seen.update(identifiers)

    def test_every_reference_family_executes_and_never_supervises_business_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            for kind in KINDS:
                rows, result = execute_reference(episode('dev',kind,2,7), Path(temporary)/kind)
                self.assertTrue(result['passed'], kind)
                for row in rows:
                    self.assertEqual([],validate_training_row(row))
                    self.assertEqual([False]*(len(row['messages'])-1)+[True],row['loss_mask'])
                    names = [call['function']['name'] for call in row['messages'][-1].get('tool_calls',[])]
                    self.assertFalse(set(names) - {'list_orders','get_order','preview_cancel','escalate_to_human'})
                if kind == 'chat_confirm': self.assertEqual(0,result['cancel_attempts'])

    def test_test_split_cannot_be_replayed_for_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'sealed_test'):
                execute_reference(episode('test','unique_exact',0,7),Path(temporary)/'sealed')
            self.assertFalse((Path(temporary)/'sealed').exists())

    def test_reference_supervision_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = episode('train','confirmed_result',0,7)
            one,_ = execute_reference(case,Path(temporary)/'one')
            two,_ = execute_reference(case,Path(temporary)/'two')
            self.assertEqual(one,two)


class PrefixValidationTests(unittest.TestCase):
    def row(self):
        return {'sample_kind':'assistant_prefix','loss_mask':[False,True], 'messages':[
            {'role':'user','content':'查询'},
            {'role':'assistant','content':'','tool_calls':[{'id':'final','function':{'name':'list_orders','arguments':{}}}]}]}

    def test_only_explicit_final_target_may_lack_future_result(self):
        row = self.row()
        self.assertEqual([],validate_training_row(row))
        ordinary = deepcopy(row); ordinary.pop('sample_kind')
        self.assertIn('tool_calls_without_response:final',validate_training_row(ordinary))
        row['loss_mask'] = [True,True]
        self.assertIn('invalid_assistant_prefix_mask',validate_training_row(row))

    def test_prefix_does_not_hide_incomplete_history_or_reused_tool_ids(self):
        row = self.row()
        earlier = deepcopy(row['messages'][-1]); earlier['tool_calls'][0]['id']='earlier'
        row['messages'].insert(1,earlier); row['loss_mask']=[False,False,True]
        self.assertIn('tool_calls_without_response:earlier',validate_training_row(row))
        earlier['tool_calls'][0]['id']='final'
        self.assertIn('reused_assistant_prefix_tool_id',validate_training_row(row))

    def test_transformers_mapping_ids_preserve_assistant_loss_alignment(self):
        class MappingTokenizer:
            def apply_chat_template(self,messages,add_generation_prompt=False,**kwargs):
                ids=[]
                for message in messages:
                    ids.extend([10 if message['role']=='user' else 20, 30, 40])
                if add_generation_prompt: ids.append(20)
                return {'input_ids':ids,'attention_mask':[1]*len(ids)}
        messages=[{'role':'user','content':'query'},{'role':'assistant','content':'response'}]
        ids,mask=render_with_assistant_mask(MappingTokenizer(),messages,loss_mask=[False,True])
        self.assertEqual([10,30,40,20,30,40],ids)
        self.assertEqual([0,0,0,0,1,1],mask)
        with self.assertRaisesRegex(ValueError,'unsupported_tokenizer_input_ids'):
            chat_template_ids(type('Bad',(),{'apply_chat_template':lambda *args,**kwargs:{'input_ids':['bad']}})(),messages)


if __name__=='__main__': unittest.main()
