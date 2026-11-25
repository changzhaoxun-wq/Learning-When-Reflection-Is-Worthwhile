import tempfile
import socket
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.llm_client import LLMResponse
from src.math_reflector_pilot import (
    REFLECTION_TEMPLATE, candidate, complete_call, free_port, validate_response, visible_answer,
)
from src.utils import read_jsonl


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return text.split()


def response(reason="stop", text="Final answer: 5"):
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    return LLMResponse(text, 10, 20, 30, metadata={
        "finish_reason": reason, "raw_usage": usage,
        "generation_attempts": [{"finish_reason": reason, "context_window_retry": None}],
    })


class MathPilotTests(unittest.TestCase):
    def test_port_probe_refuses_live_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            with patch("src.math_reflector_pilot.PORT", listener.getsockname()[1]):
                with self.assertRaisesRegex(RuntimeError, "occupied"):
                    free_port()

    def test_port_probe_allows_closed_server_connections(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen(1)
            with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                connection, _ = listener.accept()
                connection.close()
                self.assertEqual(client.recv(1), b"")
        with patch("src.math_reflector_pilot.PORT", port):
            free_port()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = Mock()

    def call(self, prompt="Question", action="direct"):
        return complete_call(self.client, Tokenizer(), self.root, "base", "id-1", action, prompt)

    def test_length_retry_cost_and_resume(self):
        self.client.generate.side_effect = [response("length"), response()]
        result = self.call()
        self.assertEqual(result["total_tokens"], 60)
        self.assertEqual(result["prompt_tokens"], 20)
        self.assertEqual(result["completion_tokens"], 40)
        self.assertEqual(result["recovered_length_calls"], 1)
        self.assertEqual([c.kwargs["max_tokens"] for c in self.client.generate.call_args_list], [4096, 8192])
        self.assertEqual(self.call(), result)
        self.assertEqual(self.client.generate.call_count, 2)

    def test_cap_failure_is_retained_and_not_repeated(self):
        self.client.generate.return_value = response("length")
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "Unresolved truncation"):
                self.call()
        records = read_jsonl(self.root / "calls/base/id-1/direct.jsonl")
        self.assertEqual(len(records), 3)
        self.assertEqual(sum(r["total_tokens"] for r in records), 90)
        self.assertEqual(self.client.generate.call_count, 3)

    def test_overflow_before_request(self):
        with self.assertRaisesRegex(RuntimeError, "no input or output cropping"):
            self.call("word " * 20000)
        self.client.generate.assert_not_called()

    def test_invalid_usage_and_stop_reason(self):
        for reason in (None, "content_filter"):
            obj = response(reason)
            with self.assertRaises(RuntimeError):
                validate_response(vars(obj))
        obj = response()
        obj.total_tokens = 31
        with self.assertRaises(ValueError):
            validate_response(vars(obj))
        obj = response()
        obj.metadata["raw_usage"] = {}
        with self.assertRaises(ValueError):
            validate_response(vars(obj))
        obj = response()
        obj.metadata["generation_attempts"][0]["context_window_retry"] = {"max_tokens": 100}
        with self.assertRaises(RuntimeError):
            validate_response(vars(obj))

    def test_changed_prompt_rejected(self):
        self.client.generate.return_value = response()
        self.call()
        with self.assertRaisesRegex(ValueError, "locked prompt"):
            self.call("Different question")
        self.assertEqual(self.client.generate.call_count, 1)

    def test_reflection_conditioning_and_total_path_cost(self):
        prompt = REFLECTION_TEMPLATE.format(question="Test", current_answer="Previous answer", round_id=2)
        self.assertIn("Previous answer", prompt)
        self.assertIn("Reflection round: 2", prompt)
        self.assertEqual(visible_answer("<think>reason</think>Final answer: 5"), "Final answer: 5")
        for text, opened in (("<think>reason", False), ("reason", True)):
            with self.assertRaises(RuntimeError):
                visible_answer(text, thinking_open=opened)
        self.client.generate.return_value = response()
        call = self.call()
        result = candidate("reflect_2", [call, call, call], "5")
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["token_cost"], 90)


if __name__ == "__main__":
    unittest.main()
