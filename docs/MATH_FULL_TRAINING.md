# 全量数学训练轨迹构建

## 固定目标与边界

目标是用少量按需反思，以较低总成本接近强模型独立作答的质量。
保持 D/R1/R2 与文本隐藏向量 Gate/Round 框架，不恢复 QA，不引入在线 verifier。
本步骤只构建数据，不自动训练策略、不读取官方测试结果调整参数。

输出根目录：`outputs/math_full_training_v1/`。

## 问题划分

- 全部 7473 题，训练 5978 题、验证 1495 题。
- 已查看的 200 题 pilot 固定放入训练部分。
- 其余题按 `sha256(20260910:id)` 确定验证集，不依据正确性抽样。
- Gate、Round 后续必须使用相同问题级清单；不能在训练脚本里再次各自随机切分。
- 检查训练/官方测试的 ID 和规范化文本无重叠。此项检查不等于语义改写去污染。
- 该训练域历史上已使用过，当前划分是后续流程固定的内部验证，不宣称完全未观察过的独立确认集。

`manifest.json`、`train_items.jsonl` 和 `validation_items.jsonl` 在首次准备时锁定。

## 生成协议

沿用已核验 pilot 的字面提示、原生 chat template、temperature=0、seed=42：

| 角色 | 模型 | thinking | GPU |
|---|---|---|---|
| Base | Qwen3-8B-Instruct 本地别名 | 关闭 | 0（2026-09-12 起续跑；早期响应在 GPU 1 生成） |
| Reflector / 独立作答对照 | Qwen3-14B | 开启 | 1、3，TP=2（2026-09-13 审计迁移） |

先完整执行 Base 阶段，关闭该服务后再加载 14B，不同时加载这两组服务。
仅调整 Base 的显卡位置，保留生成配置；不同硬件执行仍可能产生数值级差异。
端口 8014 若被其他监听服务占用则停止，不杀其他进程。
每个阶段加载前检查目标显卡余量，不足时每 60 秒等待重查。14B 当前使用 GPU 1、3，
不自动改为不同精度或更短上下文。等待状态为 `phase=waiting_for_gpu_memory`。

上下文 32768，输出预算依次 4096 / 8192 / 16384，额外预留 512 token。
每次请求实际输入加最大输出预算及安全余量必须能放入窗口，不裁剪输入。
每次长度重试先持久化记录并累计其输入、输出。自 2026-09-12 的审计修订起，
三次均以 length 结束且记录通过核验的调用会被单独登记，其他题继续生成，
而不是写成普通错误。服务异常、记录损坏、窗口溢出或异常 thinking 结束仍会停止流程。
网络失败如未返回 usage，无法追溯未知服务端开销，需单独披露。

## 候选覆盖与成本

所有题都有 Direct。训练部分仅对 Direct 错误题生成 R1/R2；验证部分所有题都有
Direct、R1、R2 和 14B 独立作答，以观测错误引入和强模型基线。

训练中 Direct 正确的记录只保存真实观测的 Direct，标记
`candidate_coverage=direct_only_known_correct` 与 `missing_actions`。
不能用这种缺少候选的训练记录评估固定反思，也不能把缺失结果补为错误或零成本。
标签按正确性优先、相同正确性选总成本最低；`reward` 中 lambda=0.0001 仅用于报告。
`state_features` 只包含问题和初始响应文本；标准答案及真实候选得分只用于离线监督。
后续 Round 训练需排除缺少反思候选的题，并披露其主要来自 Direct 错题的训练分布。

R1 包含 Base+第一轮，R2 包含 Base+第一轮+第二轮，每条路径仅计一次 Base。
`independent` 字段只计强模型自身的输入输出；如果对比“先 Base 后独立作答”，需额外加一次 Base。
选择器的本地延迟需要另测，不属于生成式 token，但不能从端到端成本中忽略。

通过完整审计的 pilot Base 200 条复用；其中 Direct 错题的已有 14B 反思也复用。
引用源文件及 SHA256，而非无依据地混入历史旧轨迹。请保留 pilot 目录以供恢复时核验。
最终审计分别记录新生成和复用调用的 token，不把复用数据误报为本轮新付费调用。

## 运行、查看和恢复

```bash
cd /path/to/adaptive_reflection
bash scripts/run_math_full_training.sh --preflight-only
bash scripts/run_math_full_training.sh
```

后台启动的日志路径与 PID 见根目录 `launcher.json`。查看：

```bash
cat outputs/math_full_training_v1/status.json
tail -n 40 outputs/math_full_training_v1/pipeline_*.log
```

同一个启动命令即可恢复；已有行和逐次调用日志会核验后复用，文件锁拒绝重复运行。
若提示协议变化，不要删除锁定文件强行继续，应查明变化并建立新版本输出目录。
ETA 是当前阶段估计，缓存复用与选择性调用会使其波动，不是整个流程的准确承诺。

过程输出为 `direct_all.jsonl`、`policy_all.jsonl` 和 `calls/`。
只有所有行完成且通过复核、没有待处理异常时，才导出 `policy_train.jsonl`、
`policy_validation.jsonl`，并写入 `completion_audit.json` 与 `status.json` 的 `stage=complete`。

下一步先核验类别分布与验证候选完整性，再训练共享验证划分的向量 Gate/Round。
不自动沿用旧训练器的随机验证切分；不根据官方测试 Oracle 选 epoch 或门限。

## 输出耗尽隔离规则（2026-09-12）

原任务完成 1855 条 Base 后，在 `gsm8k-main-train-1855` 的重复输出上耗尽
4096/8192/16384 三次预算。三次共 28885 token（输入 213、输出 28672），
原始日志和费用均保留，不追加第四次请求，不改变标准答案。

旧脚本、协议和状态归档在
`infrastructure_revisions/exhausted_output_quarantine_v1/`。迁移只允许全量调度代码、
异常记录、资源等待及 Base 显卡位置变更；模型元数据、生成/评分模块、提示、
thinking、预算、训练验证划分和已有数据均做一致性校验。

- `failures/base/<id>.json` 或 `failures/qwen14/<id>.json` 保存异常原因、调用路径、哈希和费用。
- Base 未正常完成时，反思阶段标记 `blocked`，不把截断文本当成正式初始答案。
- 一题在反思或独立作答时失败，会暂缓该题的完整候选记录；此前正常调用也保留在调用日志并计费。
- 状态分别报告 `processed`（已处理）、`completed`（正常结果）、`quarantined`（待处理），不能混用。
- 全部题处理后如仍有异常，状态为 `awaiting_failure_resolution`，不是 `complete`。
  生成 `coverage_audit.json`，分别记录新生成/复用成本、待处理 ID 和各划分原始分母。
  只导出带 `_partial` 后缀的候选文件，不作为最终验证集成绩或无截断实验结论。
- 后续需制定单独、统一的异常解决或失败计分协议；不依据结果静默删题，不在此步骤自动改变标签。

本次迁移已完成后，直接使用原运行命令即可恢复，无需反复添加迁移参数。

## 反思器显卡迁移（2026-09-13）

Base 阶段已处理全部 7473 题，正常保存 7471 题，2 题输出耗尽预算仍待处理。
原反思阶段因 GPU 2 显存不足尚未开始生成，现改用空闲余量足够的 GPU 1、3。
TP=2、显存利用率 0.46、模型、提示、thinking、输出预算和失败规则都不变。
旧源码、协议、等待状态及数据哈希保存在
`infrastructure_revisions/reflector_gpu_placement_v1/`。

显式续跑命令：

```bash
bash scripts/run_math_full_training.sh --base-gpu 0 --reflector-gpus 1,3
```

省略 `--reflector-gpus` 会读取本次已锁定的显卡配置，而不是恢复为旧的 1、2。
更改编号需要新的可审计迁移，不能直接修改协议文件规避一致性检查。
GPU 位置不同仍可能带来数值执行差异，迁移记录随实验保留。
