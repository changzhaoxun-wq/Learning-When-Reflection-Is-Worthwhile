# Full-test execution-failure accounting

## Status and scope

The original run at `outputs/math_full_test_v1` finished traversing all 1319 GSM8K
test questions, with 1315 fully completed rows and four quarantined questions.
It is **not** a completed no-truncation experiment. The original clean evaluator
and its readiness gate remain unchanged.

This separate, post-hoc addendum preserves the original artifacts. It does not
change the prompts, thinking settings, temperature, seed, output caps, trained
weights, Round model, or validation-selected Gate thresholds. Test examples are
not removed and test Oracle labels are not used for training or calibration.

## Recorded failures

| Question ID suffix | Failed action | Missing independent answer |
| --- | --- | --- |
| 22 | Independent Qwen3-14B | No; the exhausted call is recorded |
| 62 | Second reflection | Yes |
| 744 | Second reflection | Yes |
| 1082 | Second reflection | Yes |

Each exhausted logical call has three paid attempts with output caps 4096,
8192, and 16384. Inspection shows repeated reasoning; this is not evidence that
an even larger budget would yield a useful answer. No extra retry is issued for
these four exhausted calls. Only the three independent answers skipped by the
original stage's early exit are generated, using the original frozen protocol.
If a supplemental call also exhausts its budget, the same accounting rule applies.

## Scoring and costs

An exhausted action receives zero **system answer credit**, a null prediction,
and `execution_status=output_budget_exhausted`. This must not be described as a
completed but semantically incorrect answer. No number is extracted from its
unfinished reasoning. Successful calls retain the original answer extractor and
numeric equality scorer.

All 1319 questions remain in every method's denominator. Reports separate
execution failures from completed wrong answers. A sensitivity field reports
the arithmetic upper value if every selected failed action were correct; it is
not a measured result or a prediction of recovery. No fallback to an earlier
answer is silently added to the frozen deployment policy.

Costs include every recorded input/output attempt. R1 includes Direct and R1;
R2 includes Direct, R1 and R2; the independent strong baseline excludes Direct.
Actual generation spend counts shared calls once, while each simulated strategy
is charged for its own execution path. Local selector compute is not converted
to tokens. Token counts across models do not directly establish speed or monetary
savings.

## Run and resume

Audit only, without starting a model or making new requests:

```bash
cd /path/to/adaptive_reflection
bash scripts/run_math_test_failure_accounting.sh
```

Supplement missing calls and evaluate all frozen policies:

```bash
cd /path/to/adaptive_reflection
bash scripts/run_math_test_failure_accounting.sh --generate --evaluate
```

The same command resumes durable calls and CPU feature shards. It uses the
original Qwen3-14B GPU placement (0,1), waits for sufficient free memory, refuses
an occupied service port, and stops only its own server after supplementation.
Offline feature extraction uses CPU. No other user's process is stopped.

Outputs are under `outputs/math_full_test_failure_accounted_v1/`:

- `manifest.json`: hashes of original artifacts and addendum implementation.
- `calls/`: supplemental requests only.
- `policy_test_accounted.jsonl`: all action outcomes, including explicit failures.
- `accounting_audit.json`: failed actions, paid attempts, usage and call hashes.
- `evaluation/summary.json`: all 36 locked policies, 12 three-seed groups, baseline
  metrics, random-call controls, paired bootstrap and Holm-adjusted McNemar tests.
- `evaluation/REPORT.md`: failure-inclusive result tables and limitations.
- `status.json`: current stage; completion is `completed_failure_inclusive`.

`manifest.json` is locked on first run. Changed source artifacts or changed
addendum code require a separately documented revision, not silent reuse.
The original run's `awaiting_failure_resolution` status is intentionally retained.
This addendum is not evidence that the four generation failures have been fixed.

## GPU extraction revision (2026-09-18)

The three supplemental generations have completed and the owned Qwen3-14B
server has exited. At the user's request, CPU extraction was stopped with its
completed shards preserved. Resume this stage with:

```bash
cd /path/to/adaptive_reflection
python -u -m src.math_test_gpu_features --gpu 0
```

This uses a separate GPU cache, FP32 with TF32 disabled, SDPA math attention,
the same two-example batches, last-two-layer mean/last pooling and input limit.
It does not regenerate LLM answers. The new runtime provider supplies its vectors
to the unchanged offline evaluator, which runs the classifiers and statistics on
CPU. GPU/CPU vectors are not silently mixed.

`gpu_execution_revision.json` freezes the runtime amendment and saved CPU shards.
`gpu_cpu_equivalence.json` compares all available CPU rows with GPU vectors and
checks Round and all 36 locked policy actions. A numerical or action mismatch
blocks evaluation. Passing this prefix audit is not a proof of bitwise equality
on the remaining rows. The report's encoder metadata records the actual device
and audit hashes; the old source/trace/weight/threshold locks remain intact.

The initial strict check failed on 2 of 1,572,864 coordinates in the 384-row CPU
prefix. All Round and 36-policy actions matched. The failed audit is retained,
not relabeled as passed. A separate, post-hoc, answer-label-free review checks
vector-level error and action stability (`gpu_numerical_review.json`). This is
a disclosed runtime adjustment, not model/threshold selection or a claim of
bitwise equality. All 1319 GPU vectors were extracted in approximately 21 seconds.

Continue from the preserved GPU cache after this explicit numerical review:

```bash
python -u -m src.review_math_gpu_runtime \
  --root outputs/math_full_test_failure_accounted_v1
```
