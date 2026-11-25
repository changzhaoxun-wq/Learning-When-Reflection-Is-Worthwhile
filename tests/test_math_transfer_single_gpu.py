import json
from pathlib import Path
import tempfile
import unittest

from src.resume_math_transfer_single_gpu import freeze_amendment


class AmendmentTest(unittest.TestCase):
    def fixture(self, root):
        for name in ('manifest', 'policy_lock', 'generation_protocol', 'generation_preflight'):
            (root / f'{name}.json').write_text('{}')
        (root / 'run_protocol.json').write_text(json.dumps({'implementation_sha256': {},
            'roles': {'qwen14': {'model': 'Qwen3-14B', 'gpus': '0,2', 'tp': 2, 'thinking': True}}}))
        path = root / 'sample/rows/base/x.json'
        path.parent.mkdir(parents=True)
        path.write_text('{}')
        return {'sample': [{'id': 'x'}]}

    def test_lock_resume_and_base_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            items = self.fixture(root)
            a = freeze_amendment(root, '0', items)
            self.assertEqual(a['new_qwen14_role']['tp'], 1)
            self.assertEqual(a['gpu_memory_utilization'], .8)
            self.assertEqual(a, freeze_amendment(root, '0', items))
            (root / 'sample/rows/base/x.json').write_text('{"changed":true}')
            with self.assertRaises(Exception):
                freeze_amendment(root, '0', items)

    def test_preexisting_strong_call_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            items = self.fixture(root)
            path = root / 'sample/calls/qwen14/x/reflect_1.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'mix'):
                freeze_amendment(root, '0', items)


if __name__ == '__main__':
    unittest.main()
