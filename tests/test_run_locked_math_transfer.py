import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.run_locked_math_transfer import base_row, reflection_row, resolve_call, blocked_candidate


def success(text='Final answer: 2'):
    return {'answer_text': text, 'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}


def failure():
    return {'execution_status': 'output_budget_exhausted', 'prompt_tokens': 30,
            'completion_tokens': 70, 'total_tokens': 100}


class PathsTest(unittest.TestCase):
    def setUp(self):
        self.item = {'id': 'x', 'question': 'One plus one?', 'gold_answer': '2'}

    def test_success_path_sums_and_visible_input(self):
        base = base_row(self.item, lambda *args: success())
        calls = []
        def complete(action, prompt):
            calls.append((action, prompt))
            return success()
        row = reflection_row(base, complete)
        self.assertEqual([c['token_cost'] for c in row['candidate_results']], [15, 30, 45])
        self.assertEqual(row['independent']['token_cost'], 15)
        self.assertIn('Final answer: 2', calls[1][1])
        self.assertEqual([c[0] for c in calls], ['reflect_1', 'reflect_2', 'independent'])

    def test_r1_failure_blocks_r2_but_not_independent(self):
        base = base_row(self.item, lambda *args: success())
        called = []
        def complete(action, prompt):
            called.append(action)
            return failure() if action == 'reflect_1' else success()
        row = reflection_row(base, complete)
        self.assertEqual(called, ['reflect_1', 'independent'])
        self.assertEqual([c['token_cost'] for c in row['candidate_results']], [15, 115, 115])
        self.assertEqual(row['candidate_results'][2]['execution_status'], 'blocked_upstream_failure')
        self.assertIsNone(row['candidate_results'][1]['pred'])

    def test_base_failure_preserves_cost_and_denominator(self):
        base = base_row(self.item, lambda *args: failure())
        called = []
        def complete(action, prompt):
            called.append(action)
            return success()
        row = reflection_row(base, complete)
        self.assertEqual(called, ['independent'])
        self.assertFalse(row['policy_input_available'])
        self.assertEqual([c['token_cost'] for c in row['candidate_results']], [100, 100, 100])

    def test_r2_failure_retains_all_previous_calls(self):
        base = base_row(self.item, lambda *args: success())
        row = reflection_row(base, lambda a, p: failure() if a == 'reflect_2' else success())
        self.assertEqual(row['candidate_results'][2]['token_cost'], 130)
        self.assertEqual(row['independent']['correct'], 1)

    def test_unknown_runtime_error_not_scored(self):
        with tempfile.TemporaryDirectory() as d, patch('src.run_locked_math_transfer.exhausted_call', return_value=None), \
                patch('src.run_locked_math_transfer.pilot.complete_call', side_effect=RuntimeError('network')):
            with self.assertRaisesRegex(RuntimeError, 'network'):
                resolve_call(Path(d), 'base', self.item, 'direct', 'prompt', None, lambda: object())

    def test_cached_failure_does_not_start_server(self):
        with tempfile.TemporaryDirectory() as d, patch('src.run_locked_math_transfer.exhausted_call', return_value=failure()):
            def forbidden():
                self.fail('Server should not start')
            result = resolve_call(Path(d), 'base', self.item, 'direct', 'prompt', None, forbidden)
            self.assertEqual(result['execution_status'], 'output_budget_exhausted')

    def test_cannot_block_healthy_path(self):
        with self.assertRaises(ValueError):
            blocked_candidate('reflect_1', {'execution_status': 'success'})

    def test_real_call_log_resume_does_not_regenerate(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [1, 2]
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]['content']
        class Client:
            calls = 0
            def generate(self, prompt, **kwargs):
                self.calls += 1
                return SimpleNamespace(text='Final answer: 2', prompt_tokens=2,
                    completion_tokens=5, total_tokens=7, metadata={
                        'finish_reason': 'stop', 'raw_usage': {'prompt_tokens': 2,
                        'completion_tokens': 5, 'total_tokens': 7}})
        client = Client()
        with tempfile.TemporaryDirectory() as d:
            first = resolve_call(Path(d), 'base', self.item, 'direct', 'question', Tokenizer(), lambda: client)
            def forbidden():
                self.fail('Replay must not load a model')
            second = resolve_call(Path(d), 'base', self.item, 'direct', 'question', Tokenizer(), forbidden)
            self.assertEqual(first, second)
            self.assertEqual(client.calls, 1)
            with self.assertRaisesRegex(ValueError, 'locked prompt'):
                resolve_call(Path(d), 'base', self.item, 'direct', 'changed', Tokenizer(), forbidden)


if __name__ == '__main__':
    unittest.main()
