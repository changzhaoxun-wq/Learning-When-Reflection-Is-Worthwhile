import unittest

from src.analyze_math_reflector_pilot import paired


class PairedMathTests(unittest.TestCase):
    def test_identical_records(self):
        rows=[{'correct':1,'token_cost':10,'reward':.999},{'correct':0,'token_cost':20,'reward':-.002}]
        result=paired(rows,rows,iterations=100)
        self.assertEqual(result['mcnemar']['exact_two_sided_p'],1)
        self.assertEqual(result['accuracy']['ci95_lower'],0)
        self.assertEqual(result['token_cost']['ci95_upper'],0)

    def test_pair_direction_and_cost(self):
        base=[{'correct':0,'token_cost':10,'reward':-.001}]*8
        selected=[{'correct':1,'token_cost':20,'reward':.998}]*8
        result=paired(selected,base,iterations=100)
        self.assertEqual(result['mcnemar']['learned_correct_baseline_wrong'],8)
        self.assertEqual(result['mcnemar']['exact_two_sided_p'],.0078125)
        self.assertEqual(result['accuracy']['observed_difference'],1)
        self.assertEqual(result['token_cost']['observed_difference'],10)

    def test_empty_and_unpaired_inputs(self):
        for a,b in (([],[]),([{}],[])):
            with self.assertRaises(ValueError):
                paired(a,b,iterations=10)


if __name__=='__main__':
    unittest.main()
