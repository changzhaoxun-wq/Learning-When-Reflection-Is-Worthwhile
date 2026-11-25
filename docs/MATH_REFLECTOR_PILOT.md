# Math Reflector Pilot

## Scope

Keep the Base, text-vector selector, Gate--Round architecture and the three
deployment actions (direct, reflect_1, reflect_2). Do not add QA ranking,
retrieval or reinforcement learning. First check candidate quality, before
training another selector.

Preparation only:

```bash
python -m src.prepare_math_reflector_pilot
```

Outputs: outputs/math_reflector_pilot_v1/development.jsonl, manifest.json,
old_base_audit.json and prompts/. The 200-question training-side pilot is
sampled without correctness or confidence filtering. It is exploratory, not
an independent test. No LLM calls or model download are performed by this tool.

## Issues Found Before Generation

The current global math reflection v4 prompts explicitly omit the previous
answer. Keep historical artifacts unchanged and label them independent
re-solving. New response-conditioned reflection must show the Base response
in round 1 and the first revision in round 2. Use the same new template for
both the old and replacement reflector; do not attribute a simultaneous
template change solely to the model.

The old 7473-row non-thinking Base trace includes 26 length-ended calls and
28 missing final-answer markers. Do not silently drop affected questions or
treat them as complete semantic errors. Any reused Base must pass per-row
protocol/completion checks, and all compared reflectors must receive the
same saved Base output. Record any generation changes separately.

## Planned Comparisons

1. Frozen Qwen3-8B Base Direct.
2. Old DeepSeek reflector, response-conditioned one/two rounds.
3. Replacement reflector, the same response-conditioned one/two rounds.
4. Replacement model independent Direct, without seeing the Base response.

Checkpoint: Qwen/Qwen3-14B, approved by the user on 2026-09-10. Download source
is ModelScope (https://modelscope.cn/models/Qwen/Qwen3-14B), destination
Qwen/Qwen3-14B. This is a candidate to test, not
a claim of higher repair quality. Download completed on 2026-09-10. All 18
files match ModelScope SHA-256 values; eight shards contain 443 tensors and
match the weight index. The local tokenizer and non-thinking chat template
loaded successfully on CPU. Downloaded files total 29,552,612,279 bytes
(29.55 GB, approximately 27.52 GiB). Verification metadata is saved at
outputs/math_reflector_pilot_v1/model_download_verification.json.
At the download checkpoint no serving or generation had started. The download command is:

```bash
modelscope download --model Qwen/Qwen3-14B \
  --local_dir Qwen/Qwen3-14B --max-workers 4
```

The initial sampling manifest remains unchanged as a historical preparation
record; its pending-model note predates the user's approval.
Current local devices have approximately 30-32 GiB free on GPUs 0-2 after QA
shutdown; weight size alone is insufficient to guarantee single-GPU serving.
Check runtime and KV-cache headroom before selecting placement.

For the independent-answer control, report both standalone cost and the cost
of Base followed by independent answering. Generate complete coverage including
Direct-correct questions. Report input/output/retry costs, fix/harm rates,
finite-candidate Oracle and paired differences. Failures remain explicit.
Do not train a new policy until the pilot establishes useful candidate space.

## Running the Frozen Pilot

The background pipeline was launched on 2026-09-10. Artifacts live under
outputs/math_reflector_pilot_v1/run_v1/. Check status.json and launcher.json
for current progress and the pipeline log; launch alone is not a completed result.

```bash
./scripts/run_math_reflector_pilot.sh --preflight-only
./scripts/run_math_reflector_pilot.sh
```

The second command resumes validated call/stage artifacts and refuses a second
concurrent runner or a changed protocol. QA remains paused. Global prompts and
old result files remain unchanged. Do not run this command again while the
background runner is active.

The Base uses Qwen3-8B with thinking disabled. DeepSeek uses its native thinking
template; Qwen3-14B uses thinking enabled. These are disclosed model-system
settings, not a controlled comparison of parameter count alone. Both reflectors
receive the same response-conditioned task template. The independent Qwen3-14B
control uses the Base task prompt with Qwen3-14B's own native chat template.

Stages load sequentially: Base and DeepSeek on GPU 0, then Qwen3-14B with tensor
parallelism on GPUs 1 and 2. The runner only stops the service process group it
created, using port 8014. It does not restart QA or stop other users' services.
Each participating GPU has a 0.46 vLLM memory budget; eager execution, one
concurrent sequence and 1024 batched tokens limit additional memory pressure.
Successful tokenizer checks do not guarantee successful GPU initialization.

All roles use a 32768-token context. Output attempts use 4096, 8192, then 16384
tokens, restarting the same request on length endings. Before every request,
actual rendered input plus 16384 output tokens plus 512 safety tokens must fit.
Initial prompt checks passed for all 200 questions with maximum required
windows of 17077 (Base), 17070 (DeepSeek) and 17073 (Qwen3-14B). Reflection
checks occur after the preceding visible response is available.

Every returned attempt is durably saved before checking its finish reason and
API usage. Input and output costs of failed-length attempts are included in
the path total. A still-truncated cap attempt, unclosed thinking output, missing
usage, server-side budget reduction or context overflow stops the pipeline;
no incomplete final comparison is written. This is a fail-closed evaluation
rule, not a promise that models will never reach their output budget. Network
failures can have unreported server costs which cannot be reconstructed.

Nine sampling/runner tests passed. The protocol locks code and prompt hashes,
model metadata hashes and shard size/mtime. The new model additionally has the
separate full download SHA-256 verification record. No policy is trained here.
Paired statistical analysis is a follow-up after complete candidate generation.

The first service launch failed before generation because the non-activated
background environment could not find the installed ninja binary. The wrapper
now includes the vLLM environment and CUDA tool directories in PATH. Launch
records and the failed log are preserved; prompts and generation budgets did
not change. Use launcher.json for the current PID rather than an older log.

## Resume After Base Completion

All 200 Base rows completed before a port-check failure at the first model
switch. A closed server can leave TIME_WAIT connections: a plain bind probe
then reports an occupied port even without a listener. The probe now uses
SO_REUSEADDR followed by listen, matching rebinding behavior while continuing
to reject a live listener. No unrelated service is stopped or reused.

The original code and protocol are archived under
run_v1/infrastructure_revisions/port_probe_v1/ (relative to the pilot output
directory). Its audit.json verifies that only free_port changed at the AST
level; all other generation/scoring logic, prompts, budgets and model settings
remain identical. Only the infrastructure code hash in protocol.json was
updated, with before/after hashes retained. All 200 Base call logs, answers and
cumulative costs passed integrity checks and are reused. Eleven tests passed,
including live-listener rejection and closed-server rebinding.

The pipeline resumed on 2026-09-10 at 11:22 from the DeepSeek stage. This is a
launch record, not completion evidence; consult status.json and launcher.json.
QA remains outside the active experiments.

## Completed Pilot Analysis

All three stages finished with 200 aligned questions each. The model server
has stopped. One length retry recovered; zero unresolved truncations remain.
Raw candidates and gold were not modified. The reusable offline audit is:

```bash
python -m src.analyze_math_reflector_pilot --output outputs/math_reflector_pilot_v1/run_v1/analysis_recheck
```

Use a new output directory; the analyzer refuses to overwrite an existing one.
Existing results are in
outputs/math_reflector_pilot_v1/run_v1/analysis_v1/REPORT_ZH.md,
with case review and next-step constraints in INTERPRETATION_ZH.md.

Base / Qwen3-14B R1 / R2 accuracy is 91.0% / 94.0% / 94.5%, but
independent Qwen3-14B solving reaches 95.5% with lower fixed-method token
costs. These are exploratory training-side point estimates, not evidence that
reflection outperforms independent solving. The reflection Oracle is 95.5%
at 339.46 average tokens and is not deployable. Only nine questions receive a
non-Direct Oracle label; no selector was trained on this pilot. Full paired
statistics and the unchanged-gold case review must accompany interpretation.
