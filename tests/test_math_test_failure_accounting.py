import contextlib
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src import account_math_test_failures as accounting
from src import evaluate_accounted_math_test as offline
from src.evaluate_locked_math_test import assert_complete_trace


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


class AccountingTests(unittest.TestCase):
    def fixture(self, source):
        tokenizer = Tokenizer()
        items = [{'id': key, 'question': key, 'gold_answer': '1', 'policy_split': 'test'}
                 for key in ('r2_failed', 'ind_failed', 'healthy')]
        generated = []

        def generate(prompt, **kwargs):
            generated.append(prompt)
            length = ('r2_failed' in prompt and 'Reflection round: 2' in prompt) or (
                'ind_failed' in prompt and prompt.startswith('You are solving'))
            p, c = len(prompt.split()), kwargs['max_tokens'] if length else 5
            return SimpleNamespace(text='Final answer: 1', prompt_tokens=p, completion_tokens=c,
                total_tokens=p+c, metadata={'finish_reason': 'length' if length else 'stop',
                    'raw_usage': {'prompt_tokens': p, 'completion_tokens': c, 'total_tokens': p+c},
                    'request': {'temperature': 0., 'seed': 42, 'effective_max_tokens': kwargs['max_tokens']}})

        base = {}
        for item in items:
            key = item['id']
            prompt = accounting.pilot.MATH_DIRECT_PROMPT.format(question=key)
            # All initial answers complete; only the strong model has fixture failures.
            def base_generate(prompt, **kwargs):
                return SimpleNamespace(text='Final answer: 1', prompt_tokens=10, completion_tokens=5,
                    total_tokens=15, metadata={'finish_reason': 'stop',
                        'raw_usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})
            first = accounting.pilot.complete_call(SimpleNamespace(generate=base_generate), tokenizer,
                    source, 'base', key, 'direct', prompt)
            direct = accounting.pilot.candidate('direct', [first], '1')
            base[key] = {**item, 'initial_call': first, 'initial_answer': first['answer_text'],
                         'initial_pred': '1', 'candidate_results': [direct]}
            answer = first['answer_text']
            for round_id in (1, 2):
                prompt = accounting.pilot.REFLECTION_TEMPLATE.format(question=key, current_answer=answer, round_id=round_id)
                try:
                    call = accounting.pilot.complete_call(SimpleNamespace(generate=generate), tokenizer,
                        source, 'qwen14', key, f'reflect_{round_id}', prompt)
                    answer = call['answer_text']
                except RuntimeError:
                    break
            if key != 'r2_failed':
                try:
                    accounting.pilot.complete_call(SimpleNamespace(generate=generate), tokenizer,
                        source, 'qwen14', key, 'independent', accounting.pilot.MATH_DIRECT_PROMPT.format(question=key))
                except RuntimeError:
                    pass
        return tokenizer, items, base, generated

    def test_reconstruction_charges_failures_and_never_extracts_their_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            source, root = Path(directory) / 'source', Path(directory) / 'new'
            tokenizer, items, base, generated = self.fixture(source)
            root.mkdir()
            before = {str(p): accounting.digest(p) for p in source.rglob('*.jsonl')}
            # Missing independent call may not silently receive zero credit.
            with self.assertRaises(RuntimeError):
                accounting.reconstruct(items[0], base['r2_failed'], source, root, tokenizer)
            row = accounting.reconstruct(items[1], base['ind_failed'], source, root, tokenizer)
            failed = row['independent']
            self.assertEqual(failed['correct'], 0)
            self.assertIsNone(failed['pred'])
            self.assertIsNone(failed['final_answer'])
            self.assertFalse(failed['scored_as_semantic_answer'])
            self.assertEqual(failed['completion_tokens'], 4096+8192+16384)
            self.assertEqual(row['candidate_results'][2]['correct'], 1)
            self.assertEqual(row['execution_failures'], ['independent'])
            self.assertEqual(before, {str(p): accounting.digest(p) for p in source.rglob('*.jsonl')})
            # Failure-inclusive rows must never pass the original clean evaluation.
            with self.assertRaises(ValueError):
                assert_complete_trace(source, {}, {'final_evaluation_ready': False})

    def test_supplement_only_missing_independent_and_resume_no_server(self):
        with tempfile.TemporaryDirectory() as directory:
            source, root = Path(directory) / 'source', Path(directory) / 'new'
            tokenizer, items, base, _ = self.fixture(source)
            root.mkdir()
            generated = []
            @contextlib.contextmanager
            def server(folder, role):
                def generate(prompt, **kwargs):
                    generated.append(prompt)
                    self.assertIn('Question:\nr2_failed', prompt)
                    self.assertNotIn('Current answer:', prompt)
                    return SimpleNamespace(text='Final answer: 1', prompt_tokens=10, completion_tokens=5,
                        total_tokens=15, metadata={'finish_reason': 'stop',
                            'raw_usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})
                yield SimpleNamespace(generate=generate)
            with patch.object(accounting.pilot, 'server', server), patch.object(accounting, 'wait_for_resources'):
                self.assertFalse(accounting.supplement(source, root, items, tokenizer, False))
                self.assertEqual(generated, [])
                self.assertTrue(accounting.supplement(source, root, items, tokenizer, True))
            with patch.object(accounting.pilot, 'server', side_effect=AssertionError('No server on resume')):
                self.assertTrue(accounting.supplement(source, root, items, tokenizer, True))
            self.assertEqual(len(generated), 1)
            row = accounting.reconstruct(items[0], base['r2_failed'], source, root, tokenizer)
            self.assertEqual(row['execution_failures'], ['reflect_2'])
            failed = row['candidate_results'][2]
            self.assertEqual(failed['token_cost'], sum(c['total_tokens'] for c in failed['calls']))
            self.assertEqual(failed['completion_tokens'], 5+5+4096+8192+16384)
            self.assertEqual(row['independent']['correct'], 1)
            metrics = offline.failure_metrics([row], ['reflect_2'])
            self.assertEqual(metrics['execution_failure_count'], 1)
            self.assertEqual(metrics['semantic_wrong_count'], 0)
            self.assertEqual(metrics['accuracy_if_selected_failures_all_correct'], 1)
            (root / 'manifest.json').write_text('{}')
            with patch.object(accounting, 'verify_snapshot'):
                accounting.export(root, source, items, base, tokenizer, {})
                accounting.export(root, source, items, base, tokenizer, {})
            audit = json.loads((root / 'accounting_audit.json').read_text())
            self.assertEqual(audit['logical_calls'], 12)
            self.assertEqual(len(audit['unresolved_execution_failures']), 2)
            self.assertFalse(audit['clean_generation_complete'])
            accounted = accounting.read_jsonl(root / 'policy_test_accounted.jsonl')
            self.assertEqual(audit['generation_usage']['total_tokens'], sum(
                r['candidate_results'][2]['token_cost'] + r['independent']['token_cost'] for r in accounted))
            # Unused paid requests must not disappear from generation cost accounting.
            (root / 'calls/qwen14/r2_failed/extra.jsonl').write_text('{}\n')
            with patch.object(accounting, 'verify_snapshot'), self.assertRaises(ValueError):
                accounting.export(root, source, items, base, tokenizer, {})

    def test_all_36_frozen_points_reported_and_marked_failure_inclusive(self):
        rows = []
        for i in range(3):
            cs = [{'action': a, 'correct': int(i == 0 or j == 1), 'prompt_tokens': 10*(j+1),
                   'completion_tokens': 5*(j+1), 'token_cost': 15*(j+1),
                   'execution_status': 'output_budget_exhausted' if i == 1 and j == 2 else 'success'}
                  for j, a in enumerate(('direct', 'reflect_1', 'reflect_2'))]
            rows.append({'id': str(i), 'candidate_results': cs, 'independent': {
                'correct': 1, 'prompt_tokens': 12, 'completion_tokens': 8, 'token_cost': 20, 'execution_status': 'success'}})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'accounting_audit.json').write_text('{}')
            (root / 'policy_lock.json').write_text('{}')
            policies = [{'name': f'{seed}_{recipe}_{branch}_{budget}', 'seed': seed, 'recipe': recipe,
                         'branch': branch, 'validation_call_budget': budget, 'model': 'gate',
                         'epoch': 1, 'direct_threshold': .5}
                for seed, recipe, branch, budget in itertools.product((7, 21, 42), ('ce', 'ranking'),
                    ('original_round', 'fixed_r1'), (.05, .1, .2))]
            lock = {'policies': policies, 'round_model': 'round', 'encoder': {'encoder_name': 'fake', 'max_length': 2048},
                    'random_repetitions': 20, 'bootstrap_iterations': 20, 'historical_limitation': 'fixture', 'training_coverage': {}}
            audit = {'analysis_type': 'posthoc_failure_inclusive_not_clean_confirmation',
                     'unresolved_execution_failures': [{'id': '1', 'action': 'reflect_2'}], 'clean_generation_complete': False}
            def load(path):
                return {'action_labels': ['reflect_1', 'reflect_2']} if path == 'round' else {
                    'action_labels': ['direct', 'reflect_1'], 'direct_threshold': .5}
            scores = np.array([[.9, .1], [.1, .9], [.1, .9]])
            with patch.object(offline, 'validate_inputs', return_value=(root, lock, rows, audit)), \
                    patch.object(offline, 'extract_features', return_value=(np.zeros((3, 2)), {'encoder_clipped_count': 0})) as encode, \
                    patch.object(offline, 'load_pickle', side_effect=load), \
                    patch.object(offline, 'score_model', return_value=scores), \
                    patch('src.evaluate_locked_math_test.score_model', return_value=scores):
                offline.evaluate(root)
                offline.evaluate(root)
                encode.assert_called_once()
            result = json.loads((root / 'evaluation/summary.json').read_text())
            self.assertEqual(len(result['policies']), 36)
            self.assertEqual(len(result['groups']), 12)
            self.assertEqual(len(result['mcnemar_vs_direct']), 39)
            self.assertTrue(result['no_test_calibration'])
            self.assertFalse(result['clean_generation_complete'])
            self.assertFalse(result['main_model_selected'])
            self.assertTrue(all(p['test_rate_exceeds_validation_budget'] for p in result['policies']))
            self.assertEqual(result['baselines']['fixed_reflect_2']['execution_failure_count'], 1)
            self.assertEqual(json.loads((root / 'status.json').read_text())['stage'], 'completed_failure_inclusive')


if __name__ == '__main__':
    unittest.main()
