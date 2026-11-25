import json
import unittest

from src.llm_client import LLMResponse, OpenAICompatibleClient
from src.neural_policy import build_policy_text
from src.prompts import reflection_prompt
from src.reflector import run_reflection


class QANoTruncationTests(unittest.TestCase):
    def test_length_retry_counts_every_attempt(self):
        client = OpenAICompatibleClient(
            base_url="http://unused/v1",
            api_key=None,
            model_name="test",
            max_tokens=4,
            retry_on_length=True,
            max_tokens_cap=8,
            fail_on_truncation=True,
        )
        responses = [
            (
                json.dumps({
                    "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                }),
                4,
                None,
            ),
            (
                json.dumps({
                    "choices": [{"message": {"content": "Final answer: Paris"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
                }),
                8,
                None,
            ),
        ]

        def fake_request(**kwargs):
            return responses.pop(0)

        client._request_payload = fake_request
        response = client.generate("question")
        self.assertEqual(response.text, "Final answer: Paris")
        self.assertEqual(response.prompt_tokens, 6)
        self.assertEqual(response.completion_tokens, 9)
        self.assertEqual(response.total_tokens, 15)
        self.assertEqual(len(response.metadata["generation_attempts"]), 2)

    def test_context_exhaustion_fails_instead_of_writing_truncated_output(self):
        client = OpenAICompatibleClient(
            base_url="http://unused/v1",
            api_key=None,
            model_name="test",
            max_tokens=4,
            retry_on_length=True,
            max_tokens_cap=8,
            fail_on_truncation=True,
        )
        raw = json.dumps({
            "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })
        client._request_payload = lambda **kwargs: (
            raw,
            2,
            {"context_limit": 12, "input_tokens": 10, "requested_max_tokens": 4, "effective_max_tokens": 2},
        )
        with self.assertRaisesRegex(RuntimeError, "context window"):
            client.generate("question")

    def test_completion_stop_sequences_are_sent_and_recorded(self):
        client = OpenAICompatibleClient(
            base_url="http://unused/v1",
            api_key=None,
            model_name="test",
            api_mode="completions",
            max_tokens=32,
            stop=["\n\nAnswer type:"],
        )
        captured = {}
        raw = json.dumps({
            "choices": [{"text": "Final answer: Paris", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        })

        def fake_request(**kwargs):
            captured.update(kwargs["payload"])
            return raw, 32, None

        client._request_payload = fake_request
        response = client.generate("question")
        self.assertEqual(captured["stop"], ["\n\nAnswer type:"])
        self.assertEqual(response.metadata["request"]["stop"], ["\n\nAnswer type:"])
        self.assertEqual(response.metadata["finish_reason"], "stop")

    def test_qa_reflection_target_audits_previous_answer(self):
        first = reflection_prompt("Context: x\n\nQuestion: y", "wrong A", 1, "qa")
        second = reflection_prompt("Context: x\n\nQuestion: y", "wrong B", 1, "qa")
        self.assertNotEqual(first, second)
        self.assertIn("wrong A", first)
        self.assertIn("wrong B", second)
        self.assertNotIn("Answer type:", first)
        self.assertIn("exact requested answer type", first)
        self.assertIn("Decision: <KEEP or REVISE>", first)
        self.assertIn("Final answer:", first)

        audit = reflection_prompt("Context: x\n\nQuestion: y", "wrong A", 2, "qa")
        self.assertNotIn("Answer type:", audit)
        self.assertIn("wrong A", audit)
        self.assertIn("second target-aware audit", audit)
        self.assertIn("Decision: <KEEP or REVISE>", audit)
        self.assertIn("Final answer:", audit)

    def test_qa_truncated_reflection_uses_counted_independent_recovery(self):
        class ScriptedLLM:
            def __init__(self):
                self.calls = []
                self.responses = [
                    LLMResponse(
                        text="Evidence chain: repeated repeated",
                        prompt_tokens=10,
                        completion_tokens=256,
                        total_tokens=266,
                        metadata={
                            "finish_reason": "length",
                            "reported_total_tokens": 266,
                            "generation_attempts": [
                                {
                                    "attempt": 1,
                                    "finish_reason": "length",
                                    "prompt_tokens": 10,
                                    "completion_tokens": 256,
                                    "total_tokens": 266,
                                }
                            ],
                        },
                    ),
                    LLMResponse(
                        text="Paris",
                        prompt_tokens=8,
                        completion_tokens=2,
                        total_tokens=10,
                        metadata={
                            "finish_reason": "stop",
                            "reported_total_tokens": 10,
                            "generation_attempts": [
                                {
                                    "attempt": 1,
                                    "finish_reason": "stop",
                                    "prompt_tokens": 8,
                                    "completion_tokens": 2,
                                    "total_tokens": 10,
                                }
                            ],
                        },
                    ),
                ]

            def generate(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                return self.responses.pop(0)

        llm = ScriptedLLM()
        response = run_reflection(
            llm,
            "Context: France contains Paris.\n\nQuestion: What is the capital?",
            "wrong",
            1,
            task_type="qa",
        )
        self.assertEqual(response.text, "Final answer: Paris")
        self.assertEqual(response.prompt_tokens, 18)
        self.assertEqual(response.completion_tokens, 258)
        self.assertEqual(response.total_tokens, 276)
        self.assertEqual(response.metadata["finish_reason"], "stop")
        self.assertTrue(response.metadata["qa_truncation_recovery"]["applied"])
        self.assertEqual(
            [item["phase"] for item in response.metadata["generation_attempts"]],
            ["reflection_truncated", "truncation_recovery"],
        )
        self.assertEqual(len(llm.calls), 2)
        self.assertFalse(llm.calls[0][1]["fail_on_truncation"])
        self.assertTrue(llm.calls[1][1]["fail_on_truncation"])

    def test_qa_missing_final_marker_uses_counted_format_repair(self):
        class ScriptedLLM:
            def __init__(self):
                self.calls = []
                self.responses = [
                    LLMResponse(
                        text="Therefore, Paris is the answer.",
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        metadata={"finish_reason": "stop", "generation_attempts": []},
                    ),
                    LLMResponse(
                        text="Paris",
                        prompt_tokens=8,
                        completion_tokens=4,
                        total_tokens=12,
                        metadata={"finish_reason": "stop", "generation_attempts": []},
                    ),
                ]

            def generate(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                return self.responses.pop(0)

        llm = ScriptedLLM()
        response = run_reflection(llm, "Context: x\n\nQuestion: city?", "wrong", 1, task_type="qa")
        self.assertEqual(response.text, "Final answer: Paris")
        self.assertEqual(response.prompt_tokens, 18)
        self.assertEqual(response.completion_tokens, 9)
        self.assertEqual(response.total_tokens, 27)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(llm.calls[0][1]["completion_prompt_suffix"].endswith("Decision: "))
        self.assertEqual(llm.calls[1][1]["stop"], ["\n"])
        self.assertIn("Answer span:", llm.calls[1][1]["completion_prompt_suffix"])
        self.assertIn("Current answer before reflection:", llm.calls[1][0])
        self.assertIn("Final answer: wrong", llm.calls[1][0])
        self.assertTrue(response.metadata["qa_format_repair"]["applied"])

    def test_qa_abbreviation_question_uses_supported_expansion(self):
        class ScriptedLLM:
            def generate(self, prompt, **kwargs):
                return LLMResponse(
                    text=(
                        "Evidence chain: CIT Group Inc. is an abbreviation for Commercial Investment Trust.\n"
                        "Final answer: CIT Group"
                    ),
                    prompt_tokens=10,
                    completion_tokens=12,
                    total_tokens=22,
                    metadata={"finish_reason": "stop", "generation_attempts": []},
                )

        response = run_reflection(
            ScriptedLLM(),
            "Context: CIT stands for Commercial Investment Trust.\n\n"
            "Question: What does the abbreviation stand for?",
            "wrong",
            1,
            task_type="qa",
        )
        self.assertEqual(response.text, "Final answer: Commercial Investment Trust")
        self.assertEqual(response.total_tokens, 22)
        self.assertEqual(
            response.metadata["qa_format_repair"]["method"],
            "deterministic_abbreviation_expansion",
        )

    def test_qa_long_final_sentence_uses_format_repair(self):
        class ScriptedLLM:
            def __init__(self):
                self.responses = [
                    LLMResponse(
                        text="Final answer: Jonathan Stark won more Grand Slam titles than Henri Leconte.",
                        prompt_tokens=10,
                        completion_tokens=12,
                        total_tokens=22,
                        metadata={"finish_reason": "stop", "generation_attempts": []},
                    ),
                ]

            def generate(self, prompt, **kwargs):
                return self.responses.pop(0)

        response = run_reflection(
            ScriptedLLM(),
            "Context: x\n\nQuestion: Who won more?",
            "wrong",
            1,
            task_type="qa",
        )
        self.assertEqual(response.text, "Final answer: Jonathan Stark")
        self.assertEqual(response.total_tokens, 22)
        self.assertEqual(
            response.metadata["qa_format_repair"]["reason"],
            "non_minimal_final_span",
        )
        self.assertEqual(
            response.metadata["qa_format_repair"]["method"],
            "deterministic_comparison_subject",
        )

    def test_truncated_format_repair_falls_back_to_complete_draft_and_counts_tokens(self):
        class ScriptedLLM:
            def __init__(self):
                self.calls = []
                self.responses = [
                    LLMResponse(
                        text=(
                            "Evidence chain: the complete draft identifies the division.\n"
                            "Final answer: NCAA Division I Football Bowl Subdivision level for all teams"
                        ),
                        prompt_tokens=10,
                        completion_tokens=12,
                        total_tokens=22,
                        metadata={
                            "finish_reason": "stop",
                            "reported_total_tokens": 22,
                            "generation_attempts": [{
                                "attempt": 1,
                                "finish_reason": "stop",
                                "prompt_tokens": 10,
                                "completion_tokens": 12,
                                "total_tokens": 22,
                            }],
                        },
                    ),
                    LLMResponse(
                        text="repetitive " * 128,
                        prompt_tokens=8,
                        completion_tokens=256,
                        total_tokens=264,
                        metadata={
                            "finish_reason": "length",
                            "reported_total_tokens": 264,
                            "generation_attempts": [{
                                "attempt": 1,
                                "finish_reason": "length",
                                "prompt_tokens": 8,
                                "completion_tokens": 256,
                                "total_tokens": 264,
                            }],
                        },
                    ),
                ]

            def generate(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                return self.responses.pop(0)

        llm = ScriptedLLM()
        response = run_reflection(
            llm,
            "Context: x\n\nQuestion: What NCAA division?",
            "wrong",
            1,
            task_type="qa",
        )
        self.assertEqual(
            response.text,
            "Final answer: NCAA Division I Football Bowl Subdivision level for all teams",
        )
        self.assertEqual(response.prompt_tokens, 18)
        self.assertEqual(response.completion_tokens, 268)
        self.assertEqual(response.total_tokens, 286)
        self.assertEqual(response.metadata["finish_reason"], "stop")
        repair = response.metadata["qa_format_repair"]
        self.assertEqual(
            repair["method"],
            "complete_draft_extraction_after_truncated_repair",
        )
        self.assertFalse(repair["repair_output_used"])
        self.assertFalse(llm.calls[1][1]["fail_on_truncation"])


    def test_qa_policy_keeps_question_and_initial_answer_before_context(self):
        row = {
            "task_type": "qa",
            "raw_question": "Who founded the institute?",
            "context": "evidence " * 1000,
            "initial_answer": "Final answer: Ada",
        }
        text = build_policy_text(row, max_text_chars=500, include_state=False)
        self.assertTrue(text.startswith("Question:\nWho founded the institute?"))
        self.assertIn("Initial answer:\nFinal answer: Ada", text)
        self.assertLessEqual(len(text), 500)


if __name__ == "__main__":
    unittest.main()
