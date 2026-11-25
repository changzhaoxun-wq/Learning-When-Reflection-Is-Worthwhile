import unittest

from src.prepare_math_reflector_pilot import select_rows


class MathPilotSamplingTests(unittest.TestCase):
    def test_outcomes_do_not_affect_selection(self):
        train = [{"id": str(i), "question": f"question {i}", "correct": i % 2} for i in range(20)]
        first, _ = select_rows(train, [], 5, 42)
        changed = [{**r, "correct": 1 - r["correct"], "gold_answer": "changed"} for r in reversed(train)]
        second, _ = select_rows(changed, [], 5, 42)
        self.assertEqual([r["id"] for r in first], [r["id"] for r in second])

    def test_test_overlap_and_duplicate_questions_removed(self):
        train = [{"id": "a", "question": "Hello  world"}, {"id": "b", "question": "math"},
                 {"id": "c", "question": "MATH"}, {"id": "d", "question": "last"}]
        chosen, excluded = select_rows(train, [{"id": "test", "question": "HELLO world"}], 2, 42)
        self.assertEqual({r["id"] for r in chosen}, {"b", "d"})
        self.assertEqual(excluded, ["a", "c"])

    def test_insufficient_or_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            select_rows([], [], 1, 42)
        with self.assertRaises(ValueError):
            select_rows([{"id": "a", "question": "one"}, {"id": "a", "question": "two"}], [], 1, 42)


if __name__ == "__main__":
    unittest.main()
