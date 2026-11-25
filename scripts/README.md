# Scripts

本目录只保留当前论文主线和仍需维护的通用入口。所有命令默认从项目根目录执行：

```bash
cd /path/to/adaptive_reflection
```

历史 MVP、旧 policy 原型、MATH-500 微调和 LoRA 实验脚本不纳入本发布副本；
当前目录只保留可维护的主线入口。

## 当前迁移阶段（2026-09-19）

`bash scripts/resume_math_transfer_single_gpu.sh --gpu 0`
生成 GSM1K、SVAMP、MultiArith 的当前协议候选。该命令自行管理服务、顺序加载模型，
支持原目录续跑；`--preflight-only` 不调用模型。同一目录勿重复启动。
强模型现改用单卡 TP=1、显存比例 0.80；原双卡运行协议保留，修订单独登记。
当前入口只做生成与失败成本审计，不自动训练或评估策略。
详见 [迁移说明](../docs/MATH_LOCKED_TRANSFER.md)。旧 `run_cross_dataset_transfer.sh` 不适用于本轮。

## 历史数学主线（2026-09-16）

固定划分上的 Gate/Round 训练、三种子验证和随机门控对照已完成。
使用 `bash scripts/run_math_full_test.sh --base-gpu 0 --reflector-gpus 0,1`
生成 GSM8K 全部 1319 个测试问题的完整候选，并自动进行冻结策略 CPU 评估。
策略与阈值在生成前锁定，不根据测试结果重新训练或校准。
输出保存在 `outputs/math_full_test_v1/`；同一命令支持断点续跑，
添加 `--preflight-only` 只做预检查。该入口自行管理模型服务，勿重复启动。
详见 [全量测试说明](../docs/MATH_FULL_TEST.md)。QA 仍暂停。

## 全量训练阶段（2026-09-10）

200 题开发对照已经完成。使用 `bash scripts/run_math_full_training.sh`
构建全部 7473 题的训练/验证候选；添加 `--preflight-only` 只检查协议与输入窗口。
该脚本管理模型生命周期，无需另启 stage2 服务。QA 暂停，不执行下面的 QA 入口。
详见 [全量训练说明](../docs/MATH_FULL_TRAINING.md)。
训练/验证清单已统一；后续训练前必须适配这份固定划分，不能直接沿用旧脚本内部随机切分。

## 历史通用流程

1. 启动模型服务：`start_stage2_vllm_shared.sh`（短上下文数学）或
   `start_stage2_vllm_dedicated.sh`（长上下文/独占显卡）。
2. 生成 direct-first 轨迹：数学使用 `run_gsm8k_direct_then_reflect.sh`，QA 使用 `run_qa_direct_then_reflect.sh`。
3. 生成分层数据：同一脚本使用 `PHASE=prepare`，或单独使用
   `prepare_hierarchical_policy_dataset.sh`。
4. 训练单个策略：`run_train_transformer_policy.sh`。
5. 多随机种子训练：`run_hierarchical_multiseed.sh`。
6. 离线评估：`run_eval_policy_offline.sh` 或 `run_final_paper_eval.sh`。
7. 扩展实验：跨数据集、lambda、解耦和显著性脚本。

## 轨迹与策略

| 脚本 | 用途 |
| --- | --- |
| `run_math_full_test.sh` | 当前锁定策略全量测试入口；完整候选、截断与成本审计、自动 CPU 离线评估。 |
| `run_gsm8k_direct_then_reflect.sh` | 历史 GSM8K 通用入口；支持 direct、选择性反思、测试轨迹、prepare、resume 和 `TARGET_TRAIN` 定向修复。 |
| `run_qa_direct_then_reflect.sh` | HotpotQA、2WikiMultiHopQA、MuSiQue 主入口；包含 8k 窗口预检、direct-first、截断自动重试、严格审计、选择性训练反思和全量评估候选。 |
| `relabel_qa_traces_offline.sh` | 对已保存 HotpotQA 候选做保守 aliases/数值归一化 F1 重标注、异常 gold 隔离和分层数据重建；不调用 LLM。 |
| `prepare_hierarchical_policy_dataset.sh` | 将候选轨迹拆为 direct/reflect gate 与 reflect_1/reflect_2 round 数据。 |
| `run_train_transformer_policy.sh` | 使用小型 Transformer 隐藏层向量训练策略分类器并保存 epoch 曲线。 |
| `run_hierarchical_multiseed.sh` | 对分层策略执行多随机种子训练、阈值校准和离线测试。 |
| `run_hierarchical_lambda_sweep.sh` | 对分层策略做 token-cost lambda 消融。 |
| `run_eval_policy_offline.sh` | 在冻结候选轨迹上比较 policy，不发起 LLM 调用。 |
| `run_final_paper_eval.sh` | 使用固定多随机种子模型生成论文主结果汇总。 |

`run_gsm8k_direct_then_reflect.sh` 的常用阶段：

```bash
PHASE=direct_train ./scripts/run_gsm8k_direct_then_reflect.sh
PHASE=train_reflect ./scripts/run_gsm8k_direct_then_reflect.sh
PHASE=prepare ./scripts/run_gsm8k_direct_then_reflect.sh
PHASE=direct_test ./scripts/run_gsm8k_direct_then_reflect.sh
PHASE=test_reflect ./scripts/run_gsm8k_direct_then_reflect.sh
```

复用旧候选且只重建截断候选：

```bash
ROOT=outputs/new_run \
DIRECT_TRAIN=outputs/old_run/direct_train.jsonl \
TARGET_TRAIN=outputs/old_run/policy_train_selective.jsonl \
PHASE=train_reflect \
./scripts/run_gsm8k_direct_then_reflect.sh
```


### QA 常用阶段

```bash
DATASET=hotpotqa PHASE=preflight ./scripts/run_qa_direct_then_reflect.sh
DATASET=hotpotqa PHASE=direct_train ./scripts/run_qa_direct_then_reflect.sh
DATASET=hotpotqa PHASE=train_reflect ./scripts/run_qa_direct_then_reflect.sh
DATASET=hotpotqa PHASE=prepare ./scripts/run_qa_direct_then_reflect.sh
DATASET=hotpotqa PHASE=direct_eval ./scripts/run_qa_direct_then_reflect.sh
DATASET=hotpotqa PHASE=eval_reflect ./scripts/run_qa_direct_then_reflect.sh
```

已有完整候选轨迹时，可直接重标注并重建 policy 数据，不会调用 LLM：

```bash
RUN_ROOT=outputs/qa_direct_first/hotpotqa_8000_v12 \
./scripts/relabel_qa_traces_offline.sh
```

QA 默认 `TRAIN_LIMIT=8000`、评估集全量；`TRAIN_LIMIT=` 表示完整训练集。Reflector
默认在第二个重复的 `Evidence chain:` 或 `Evidence audit:` 答案块开始前停止；可用 JSON 格式的
`REFLECTOR_STOP` 覆盖。生成阶段若最终输出仍为 `finish_reason=length` 会立即失败，不会写入截断答案。审计报告位于
`outputs/qa_direct_first/<dataset>/audits/`。

## 跨数据集与数据准备

| 脚本 | 用途 |
| --- | --- |
| `run_cross_dataset_transfer.sh` | GSM1K、SVAMP、MultiArith 的冻结策略迁移评估。 |
| `prepare_extra_math_datasets.sh` | 整理额外数学数据集。 |
| `prepare_reasoning_datasets.sh` | 整理 GSM-Hard、MATH、HotpotQA、2WikiMultiHopQA、MuSiQue 等推理数据。 |

## 解耦与统计实验

| 脚本 | 用途 |
| --- | --- |
| `run_decoupled_ablation.sh` | 小规模 Base/Reflector 组合消融。 |
| `run_decoupled_full_multiseed.sh` | Qwen3 Base 下不同 Reflector 的全量多随机种子实验。 |
| `run_base_decoupling_full_multiseed.sh` | 替换 Base 模型的全量多随机种子实验。 |
| `run_strict_reflector_decoupling.sh` | 冻结 Base 轨迹后仅替换 Reflector 的严格解耦。 |
| `train_eval_decoupled_policy.sh` | 为每个解耦 case 训练并评估分层 policy。 |
| `run_decoupling_significance.sh` | paired bootstrap、McNemar 和 hard split 统计。 |
| `run_fair_direct_ablation.sh` | 在统一 prompt/config 下比较不同 Base 的 direct 能力。 |

## 模型服务

| 脚本 | 用途 |
| --- | --- |
| `start_stage2_vllm_shared.sh` | 短上下文数学实验的共享显卡配置。 |
| `start_stage2_vllm_dedicated.sh` | 长上下文 QA 的独占显卡配置；默认 `START_VERIFIER=0`，只启动 Base 与 Reflector。 |
| `start_stage2_vllm.sh` | 被上述两个 wrapper 调用的底层三模型启动器。 |
| `stop_stage2_vllm.sh` | 按 PID/端口停止三模型服务。 |

## 维护约定

HotpotQA 独立开发集入口：`run_qa_hard_development.sh`。支持 `prepare`、
`check`、`preflight`、`direct`、`v5`、`v7`、`audit`、`all`；只处理新预留的
development，不评估 confirmation。详细说明见 `docs/QA_HARD_DEVELOPMENT.md`。

- 新的一键脚本保留在 `scripts/` 顶层，并在本文件登记。
- 一次性脚本和被替代入口不纳入发布副本。
- 当前脚本不得依赖未发布的本机目录。
- 轨迹生成脚本必须记录 prompt、模型、API 模式和 token 上限；协议变化使用新 `ROOT`。
