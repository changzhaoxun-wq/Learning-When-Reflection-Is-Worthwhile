"""Explicit TP=1 runtime amendment; preserves existing Base and original protocol."""
import argparse
import contextlib
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from . import run_locked_math_transfer as transfer
from . import math_reflector_pilot as pilot
from .prepare_qa_hard_development import file_hash
from .train_fixed_math_policy import write_json

GPU_UTILIZATION = .80


def preserved_files(root):
    paths = [root / name for name in ('manifest.json', 'policy_lock.json', 'generation_protocol.json',
                                     'run_protocol.json', 'generation_preflight.json')]
    paths += sorted(root.glob('*/test_items.jsonl'))
    paths += sorted(root.glob('*/rows/base/*.json'))
    paths += sorted(root.glob('*/calls/base/*/*.jsonl'))
    return {str(p): file_hash(p) for p in paths}


def freeze_amendment(root, gpu, items):
    original = json.loads((root / 'run_protocol.json').read_text())
    for name, sha in original['implementation_sha256'].items():
        if file_hash(pilot.PROJECT / 'src' / name) != sha:
            raise ValueError(f'Original implementation changed: {name}')
    for dataset, rows in items.items():
        paths = list((root / dataset / 'rows/base').glob('*.json'))
        if {p.stem for p in paths} != {r['id'] for r in rows}:
            raise ValueError('Require full Base coverage before single-GPU migration')
    path = root / 'single_gpu_amendment.json'
    if not path.exists() and (list(root.glob('*/calls/qwen14/*/*.jsonl')) or
                              list(root.glob('*/rows/qwen14/*.json'))):
        raise ValueError('Refusing to mix pre-amendment strong-model calls')
    amendment = {'reason': 'User approved single-GPU continuation because second card is unavailable',
        'preserved_files_sha256': preserved_files(root),
        'implementation_sha256': file_hash(Path(__file__)),
        'launcher_sha256': file_hash(pilot.PROJECT / 'scripts/resume_math_transfer_single_gpu.sh'),
        'old_qwen14_role': original['roles']['qwen14'],
        'new_qwen14_role': {**original['roles']['qwen14'], 'gpus': gpu, 'tp': 1},
        'gpu_memory_utilization': GPU_UTILIZATION,
        'unchanged': ['model weights', 'prompts', 'thinking', 'output budgets', 'context length',
                      'temperature', 'request seed', 'scoring', 'policy weights and thresholds', 'Base calls'],
        'limitation': 'TP=2 to TP=1 can change numerical outputs; hardware/runtime equivalence is not claimed',
        'original_run_protocol_retained': True,
        'resume_entrypoint': 'scripts/resume_math_transfer_single_gpu.sh'}
    pilot.locked_json(path, amendment)
    return amendment


def wait_single(root, role):
    if role != 'qwen14':
        raise ValueError('Single-GPU migration may not regenerate Base')
    gpu = pilot.ROLES[role]['gpus']
    while True:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.free',
                              '--format=csv,noheader,nounits'], check=True, capture_output=True,
                             text=True, timeout=15).stdout
        mem = {v[0].strip(): (int(v[1]), int(v[2])) for line in out.splitlines() if (v := line.split(','))}
        total, free = mem[gpu]
        required = int(total * GPU_UTILIZATION) + 1024
        if free >= required:
            return
        status = json.loads((root / 'status.json').read_text())
        write_json(root / 'status.json', {**status, 'phase': 'waiting_for_single_gpu_memory',
                   'gpu': gpu, 'free_mib': free, 'required_mib': required, 'heartbeat_unix': time.time()})
        print(f'[single-gpu] waiting GPU {gpu}: free={free}, required={required} MiB', flush=True)
        time.sleep(60)


@contextlib.contextmanager
def single_server(root, role):
    if role != 'qwen14':
        raise ValueError('Do not regenerate Base')
    pilot.free_port()
    spec = pilot.ROLES[role]
    log = root / f'vllm_qwen14_tp1_{datetime.datetime.now():%Y%m%d_%H%M%S}.log'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = spec['gpus']
    command = [pilot.VLLM_PYTHON, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(pilot.MODELS / spec['model']), '--served-model-name', spec['model'],
        '--host', '127.0.0.1', '--port', str(pilot.PORT), '--max-model-len', str(pilot.CONTEXT),
        '--gpu-memory-utilization', str(GPU_UTILIZATION), '--enforce-eager', '--max-num-seqs', '1',
        '--max-num-batched-tokens', '1024', '--tensor-parallel-size', '1']
    with log.open('x') as handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                    stderr=subprocess.STDOUT, env=env, start_new_session=True)
    write_json(root / 'active_server.json', {'pid': process.pid, 'pgid': process.pid,
               'role': role, 'command': command, 'log': str(log), 'tp': 1})
    print(f'[single-gpu] loading Qwen3-14B GPU={spec["gpus"]} TP=1 log={log}', flush=True)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'vLLM exited {process.returncode}; inspect {log}')
            try:
                with opener.open(f'http://127.0.0.1:{pilot.PORT}/v1/models', timeout=5) as response:
                    models = json.load(response)['data']
                if any(m['id'] == spec['model'] and int(m.get('max_model_len', 0)) >= pilot.CONTEXT for m in models):
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(5)
        else:
            raise RuntimeError(f'Service readiness timed out: {log}')
        yield pilot.OpenAICompatibleClient(f'http://127.0.0.1:{pilot.PORT}/v1', 'EMPTY', spec['model'],
                     temperature=0., seed=42, timeout=1800, max_retries=0, api_mode='completions')
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
        write_json(root / 'active_server.json', {'pid': process.pid, 'role': role,
                   'tp': 1, 'stopped': True, 'exit_code': process.returncode})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=transfer.DEFAULT_ROOT)
    parser.add_argument('--gpu', default='0', choices=[str(i) for i in range(4)])
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    with (root / 'pipeline.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, items = transfer.verify_inputs(root)
        amendment = freeze_amendment(root, args.gpu, items)
        from transformers import AutoTokenizer
        tokenizers = {r: AutoTokenizer.from_pretrained(str(pilot.MODELS / pilot.ROLES[r]['model']),
                      local_files_only=True) for r in ('base', 'qwen14')}
        original = json.loads((root / 'run_protocol.json').read_text())
        transfer.preflight(root, protocol, items, original['roles']['base']['gpus'],
                           original['roles']['qwen14']['gpus'], tokenizers)
        if args.preflight_only:
            print('[single-gpu] amendment and all preserved hashes verified; no generation', flush=True)
            return
        old_server, old_wait = pilot.server, transfer.wait_for_resources
        old_role = dict(pilot.ROLES['qwen14'])
        try:
            # Reuse the frozen runner; only process-local service/resource adapters change.
            pilot.server, transfer.wait_for_resources = single_server, wait_single
            transfer.run_stage(root, 'base', items, tokenizers['base'])
            pilot.ROLES['qwen14'].update(amendment['new_qwen14_role'])
            transfer.run_stage(root, 'qwen14', items, tokenizers['qwen14'])
            pilot.ROLES['qwen14'].update(old_role)
            transfer.verify_inputs(root)
            if preserved_files(root) != amendment['preserved_files_sha256']:
                raise ValueError('Preserved Base/protocol artifacts changed')
            transfer.export(root, items)
            pilot.locked_json(root / 'single_gpu_export_provenance.json', {
                'amendment_sha256': file_hash(root / 'single_gpu_amendment.json'),
                'accounting_audits_sha256': {name: file_hash(root / name / 'accounting_audit.json') for name in items},
                'preserved_base_verified': True, 'qwen14_tp': 1})
        except BaseException as exc:
            write_json(root / 'status.json', {'stage': 'stopped_resume_required', 'tp': 1,
                       'error_type': type(exc).__name__, 'error': str(exc), 'heartbeat_unix': time.time()})
            raise
        finally:
            pilot.server, transfer.wait_for_resources = old_server, old_wait
            pilot.ROLES['qwen14'].update(old_role)


if __name__ == '__main__':
    main()
