import copy
import itertools
import unittest

from src.summarize_math_gate_multiseed import (BRANCHES, BUDGETS, METRICS, REQUIRED_CHECKS,
                                             SEEDS, TRIALS, aggregate, mean_std, validate_runs)


def fixture():
    runs = []
    for i, seed in enumerate(SEEDS):
        points = []
        for recipe, branch, budget in itertools.product(TRIALS, BRANCHES, BUDGETS):
            correct = 900 + i + (i if recipe == TRIALS[1] else 0)
            metrics = {k: 0. for k in METRICS}
            metrics.update(n=1000, correct=correct, accuracy=correct / 1000,
                           avg_token_cost=300 + i, avg_prompt_tokens=100,
                           avg_completion_tokens=200 + i, reflection_rate=budget)
            points.append({'recipe': recipe, 'branch': branch, 'budget': budget,
                           'epoch': i + 1, 'metrics': metrics})
        checks = {k: True for k in REQUIRED_CHECKS}
        runs.append({'protocol': {'seed': seed, 'source_code_sha256': f'code_{seed}',
                                  'feature_fingerprint': 'fixed', 'test_evaluated': False},
                     'summary': {'coverage': {'train': 5976, 'validation': 1492},
                                 'baselines': {'direct': {'n': 1000},
                                               'independent': {'accuracy': .975, 'avg_token_cost': 641}},
                                 'comparisons': points, 'test_evaluated': False,
                                 'accuracy_tolerance': None, 'main_model_selected': False,
                                 'control_reproduces_previous_six_points': True, **checks},
                     'status': {'stage': 'completed', 'test_evaluated': False, **checks}})
    return runs


class MultiseedTests(unittest.TestCase):
    def test_sample_standard_deviation(self):
        self.assertEqual(mean_std([1, 2, 3]), {'mean': 2., 'std': 1.})
        for values in ([1], [1, float('nan')]):
            with self.assertRaises(ValueError):
                mean_std(values)

    def test_complete_grid_and_paired_seed_differences(self):
        out = aggregate(fixture()[::-1])
        self.assertEqual(len(out['groups']), 12)
        self.assertEqual(len(out['paired_deltas']), 6)
        diff = out['paired_deltas'][0]
        self.assertEqual(diff['ranking_minus_ce']['correct'], {'mean': 1., 'std': 1.})
        self.assertEqual([d['seed'] for d in diff['per_seed']], list(SEEDS))
        self.assertFalse(out['main_model_selected'])

    def test_duplicate_seed_rejected(self):
        runs = fixture()
        runs[1]['protocol']['seed'] = 7
        with self.assertRaises(ValueError):
            validate_runs(runs)

    def test_changed_protocol_and_coverage_rejected(self):
        for change in ('feature', 'coverage', 'baseline'):
            runs = fixture()
            if change == 'feature':
                runs[1]['protocol']['feature_fingerprint'] = 'changed'
            elif change == 'coverage':
                runs[1]['summary']['coverage']['train'] -= 1
            else:
                runs[1]['summary']['baselines']['independent']['avg_token_cost'] += 1
            with self.assertRaises(ValueError):
                validate_runs(runs)

    def test_missing_or_duplicate_budget_rejected(self):
        for duplicate in (False, True):
            runs = fixture()
            points = runs[0]['summary']['comparisons']
            points.pop()
            if duplicate:
                points.append(copy.deepcopy(points[0]))
            with self.assertRaises(ValueError):
                validate_runs(runs)

    def test_failed_audit_and_unfinished_run_rejected(self):
        for key in REQUIRED_CHECKS + ('stage',):
            runs = fixture()
            runs[0]['status'][key] = False
            with self.assertRaises(ValueError):
                validate_runs(runs)

    def test_over_budget_rejected(self):
        runs = fixture()
        runs[0]['summary']['comparisons'][0]['metrics']['reflection_rate'] = .9
        with self.assertRaises(ValueError):
            validate_runs(runs)

    def test_test_evaluation_and_model_promotion_rejected(self):
        for field in ('test_evaluated', 'main_model_selected'):
            runs = fixture()
            runs[0]['summary'][field] = True
            with self.assertRaises(ValueError):
                validate_runs(runs)


if __name__ == '__main__':
    unittest.main()
