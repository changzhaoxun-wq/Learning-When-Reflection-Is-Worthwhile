import unittest

from src.prepare_locked_math_transfer import audit_rows


def row(key, question, gold='1'):
    return {'id': key, 'question': question, 'gold_answer': gold}


class AuditTests(unittest.TestCase):
    def test_overlap_duplicates_preserve_full_set(self):
        rows = [row('a', 'Hello  World'), row('b', 'hello world'), row('c', 'Other')]
        audit = audit_rows(rows, [row('train', 'HELLO WORLD')])
        self.assertEqual(audit['subsets']['full'], ['a', 'b', 'c'])
        self.assertEqual(audit['subsets']['unique_question'], ['a', 'c'])
        self.assertEqual(audit['subsets']['nonoverlap_unique'], ['c'])
        self.assertEqual(audit['training_validation_overlap'], {'a': ['train'], 'b': ['train']})

    def test_conflicting_gold_reported(self):
        audit = audit_rows([row('a', 'Same', '1'), row('b', 'same', '2')], [])
        self.assertEqual(audit['conflicting_gold_groups'], [['a', 'b']])

    def test_invalid_inputs(self):
        for rows in ([], [row('a', '')], [row('a', 'x'), row('a', 'y')]):
            with self.assertRaises(ValueError):
                audit_rows(rows, [])


if __name__ == '__main__':
    unittest.main()
