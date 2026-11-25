# Adaptive Reflection

可复现的“大语言模型自适应反思策略学习”代码仓库。项目学习一个轻量策略，先根据问题和 Direct 初始回答判断是否值得调用反思，再决定采用一次还是两次反思，以在答案质量和生成成本之间取得平衡。

This repository contains the implementation of adaptive reflection policy learning for mathematical reasoning. The policy uses the question and the initial answer to decide whether reflection is needed and, when triggered, whether to use one or two reflection rounds.

## Repository scope

The GitHub copy contains source code, experiment entry points, configuration templates, documentation, paper source, and tests. It intentionally excludes:

- raw datasets and private/local model checkpoints;
- `outputs/`, generated traces, feature caches, checkpoints, and logs;
- compiled paper PDFs/DOCX files and Python bytecode;
- machine-specific experiment results.

All generated artifacts should be written to a local `outputs/` directory, which is ignored by Git. The original project directory is not required after this copy has been created.

## Quick start: no model or dataset required

The smoke test uses the built-in deterministic fake LLM and two toy examples embedded in the loader. It exercises the complete lightweight path:

1. construct Direct/R1/R2 candidate rows;
2. train a policy classifier;
3. evaluate Direct, fixed reflection, and learned-policy actions;
4. write and validate JSON/PKL outputs in a temporary directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/smoke_test.sh
```

The smoke test does not create repository files and does not call a network service.

## Tests and static checks

```bash
python -m pytest -q
for f in scripts/*.sh scripts/deprecated/*.sh; do bash -n "$f"; done
python -m compileall -q src tests
```

Some tests cover optional model/data integrations and may be skipped or require the corresponding dependency. The smoke test is the minimal release gate.

## Main experiment flow

The current research pipeline is divided into generation, representation extraction, policy training, and offline evaluation. Generated data are deliberately not committed:

```text
external JSONL data + model services
              |
              v
      direct / reflection traces
              |
              v
      hierarchical policy dataset
              |
              v
      Gate and Round training
              |
              v
      frozen-trace offline evaluation
              |
              v
     accuracy / token / fix-harm reports
```

For a real run, first provide local JSONL data and model endpoints. The project uses OpenAI-compatible vLLM services for generation and Hugging Face Transformers for encoder representations. The exact model revision, decoding settings, GPU placement, and context length must be recorded in the run directory rather than assumed from an alias.

### Legacy/general three-action pipeline

This path can be used with a local GSM8K-style JSONL file:

```bash
source scripts/env.example
export TRAIN_DATA=/path/to/train.jsonl
export TEST_DATA=/path/to/test.jsonl

PHASE=direct_train ROOT=outputs/gsm8k_run \
  TRAIN_DATA="$TRAIN_DATA" TEST_DATA="$TEST_DATA" \
  bash scripts/run_gsm8k_direct_then_reflect.sh

PHASE=train_reflect ROOT=outputs/gsm8k_run \
  TRAIN_DATA="$TRAIN_DATA" TEST_DATA="$TEST_DATA" \
  bash scripts/run_gsm8k_direct_then_reflect.sh

PHASE=prepare ROOT=outputs/gsm8k_run \
  TRAIN_DATA="$TRAIN_DATA" TEST_DATA="$TEST_DATA" \
  bash scripts/run_gsm8k_direct_then_reflect.sh
```

The generation scripts require running OpenAI-compatible model services. To avoid mixing protocols, use a new `ROOT` whenever prompt, model, tokenizer, or decoding configuration changes.

### Current mathematical mainline

The maintained wrappers are:

| Stage | Entry point |
| --- | --- |
| full training/validation traces | `scripts/run_math_full_training.sh` |
| locked GSM8K test traces | `scripts/run_math_full_test.sh` |
| layer/reflect-budget ablation | `scripts/run_layer_ablation_budget_math.sh` |
| frozen-policy transfer | `scripts/run_locked_math_transfer.sh` |
| offline policy evaluation | `scripts/run_eval_policy_offline.sh` |

These stages need external datasets, model checkpoints, CUDA, and vLLM/Transformers environments. They are included unchanged in scope and should be configured through their documented command-line arguments or environment variables; no local checkpoint path is assumed by this README.

### CPU/offline policy stages

Once traces and feature caches have been produced, policy training and evaluation do not require the large generator or reflector to remain loaded. Examples:

```bash
bash scripts/run_fixed_math_policy_cpu.sh
bash scripts/run_eval_policy_offline.sh
```

The input trace/cache paths must point to artifacts generated in the current run. Do not copy those artifacts into Git.

## Data contract

For the math pipeline, a minimal JSONL item is:

```json
{"id": "0", "question": "...", "answer": "reasoning #### 42"}
```

The loader also accepts `problem`/`solution` and normalized `gold_answer` fields. QA-style items may provide `context`, `answers`, and `task_type: "qa"`. See `src/gsm8k_utils.py` for normalization and `src/answer_extractor.py` for scoring behavior.

## Reproducibility rules

- Keep Direct, Reflector, encoder, tokenizer, prompt, and decoding configurations in a run manifest.
- Fit scalers and choose checkpoints/thresholds on the training/validation split only.
- Do not select thresholds or checkpoints from the test set.
- Count all actual generation input/output tokens, including retries and repair calls.
- Report Direct accuracy, final accuracy, reflection rate, average tokens, and fix/harm counts together.
- Treat `outputs/` as disposable run state, not source code.

## License and model/data terms

This repository does not redistribute model weights or benchmark data. Before public release, add the intended software license and retain the original licenses/terms for every external dataset, model, and dependency used in an experiment.
