from itertools import product
import unittest

from src.analyze_math_policy_errors import classify_case, summarize
from tests.test_fixed_math_policy import row


class ErrorAnalysisTests(unittest.TestCase):
    def test_all_correctness_gate_round_combinations(self):
        rows, cases = [], []
        for index, values in enumerate(product((0, 1), repeat=5)):
            d, r1, r2, trigger, use_r2 = values
            r = row(str(index), (d, r1, r2))
            rows.append(r)
            c = classify_case(r, bool(trigger), 'reflect_2' if use_r2 else 'reflect_1')
            cases.append(c)
            expected = r2 if trigger and use_r2 else r1 if trigger else d
            self.assertEqual(c['final_correct'], bool(expected))
            self.assertEqual(c['flags']['repairable'], not d and bool(r1 or r2))
        result = summarize(rows, cases)
        self.assertTrue(result['identities_verified'])
        self.assertEqual(result['n'], 32)
        self.assertEqual(sum(result['categories'].values()), 32)

    def test_exclusive_loss_partition_and_counterfactuals(self):
        specs = [((0, 1, 0), False, 'reflect_2'), ((0, 0, 1), True, 'reflect_1'),
                 ((0, 0, 0), True, 'reflect_2'), ((1, 0, 1), True, 'reflect_1'),
                 ((1, 0, 0), True, 'reflect_2'), ((1, 1, 1), True, 'reflect_2'),
                 ((0, 1, 1), True, 'reflect_1')]
        rows = [row(str(i), s[0]) for i, s in enumerate(specs)]
        cases = [classify_case(r, s[1], s[2]) for r, s in zip(rows, specs)]
        result = summarize(rows, cases)
        c = result['counts']
        self.assertEqual([c[k] for k in ('repairable', 'fixed', 'gate_miss', 'round_miss', 'introduced_error')], [3, 1, 1, 1, 2])
        self.assertEqual(result['oracle_gap_questions'], 4)
        self.assertEqual(c['triggered_unrepairable'], 1)
        self.assertEqual(c['gate_miss_round_would_fail'], 1)
        self.assertEqual(c['harm_avoidable_by_round'], 1)
        self.assertEqual(c['harm_unavoidable_after_trigger'], 1)
        self.assertEqual(result['counterfactuals_not_deployable']['fixed_gate_oracle_round']['correct'], 4)
        self.assertEqual(result['counterfactuals_not_deployable']['fixed_round_oracle_gate']['correct'], 4)

    def test_both_wrong_not_round_selection_error(self):
        c = classify_case(row('a', (0, 0, 0)), True, 'reflect_2')
        self.assertTrue(c['flags']['triggered_unrepairable'])
        self.assertFalse(c['flags']['round_miss'])
        self.assertEqual(c['second_round_kind'], 'both_wrong')

    def test_second_round_same_correctness_is_cost_not_answer_error(self):
        r = row('a', (1, 1, 1))
        c = classify_case(r, True, 'reflect_2')
        self.assertTrue(c['flags']['nonbeneficial_second_round'])
        self.assertFalse(c['flags']['introduced_error'])
        self.assertEqual(c['second_round_extra_tokens_vs_saved_r1'], 20)
        s = summarize([r], [c])
        self.assertEqual(s['second_round']['both_correct']['n'], 1)
        self.assertEqual(s['metrics']['correct'], s['same_gate_fixed_r1']['correct'])

    def test_missing_candidates_rejected(self):
        r = row('a')
        r['candidate_results'] = r['candidate_results'][:1]
        with self.assertRaisesRegex(ValueError, 'Complete observed'):
            classify_case(r, False, 'reflect_1')

    def test_invalid_round_and_unaligned_rows_rejected(self):
        r = row('a')
        with self.assertRaises(ValueError):
            classify_case(r, False, 'direct')
        with self.assertRaises(ValueError):
            summarize([r], [classify_case(row('b'), False, 'reflect_1')])
        with self.assertRaises(ValueError):
            summarize([], [])


if __name__ == '__main__':
    unittest.main()
