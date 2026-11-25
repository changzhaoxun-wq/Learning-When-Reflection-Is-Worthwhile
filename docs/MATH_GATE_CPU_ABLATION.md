# 纯向量 Gate 的 CPU 开发对照

这一步接在固定划分的首次 Gate/Round 训练之后，**不重新提取向量，不调用 LLM，不加载编码器，不使用 GPU**。不修改训练标签、问题划分、生成模型或提示模板，不读取独立测试集，也不覆盖此前检查点。

用户暂不设置相对强模型的精度容差，因此本实验报告差值和成本曲线，**不自动选择论文主模型，也不宣布某个结果已经达标**。

## 对照设计

所有新 Gate 使用同一训练侧 StandardScaler、原始 4096 维纯文本向量和相同 seed 42。输入依旧只有问题及初始回答。编码器全部冻结；本轮仅加载它之前生成的缓存。

| 配方 | 分类头 | 参数量 | Dropout | 正类权重 | Weight decay |
| --- | --- | ---: | ---: | ---: | ---: |
| small_unweighted | 4096 -> 16 -> 2，ReLU | 65586 | 0.2 | 1 | 0.01 |
| small_sqrt_weighted | 4096 -> 16 -> 2，ReLU | 65586 | 0.2 | min(5, sqrt(负例/正例)) | 0.01 |
| linear_sqrt_weighted | 4096 -> 2 | 8194 | 0 | 同上 | 0.01 |

AdamW，学习率 0.0003，batch size 128，每组 10 epoch。原 Gate 为 263282 参数；本轮没有扩大模型。加权项仅在训练 loss 中生效，记录的训练/验证交叉熵不加权，便于对照。正负例比例只从训练题计算。

每组 Gate 同时比较两种反思分支：

- `original_round`：固定首次训练的 Round 检查点，以同一初始状态选择 R1/R2。
- `fixed_r1`：Gate 触发后固定执行 R1。这是相同 Gate 的轮次消融，没有另行生成二分类标签或重训一个不同 Gate。

另报告原 Gate 在同样预算校准规则下的结果，以及原 Gate/Round 未改阈值的历史结果。固定旧 Round 用于隔离 Gate 改动；本轮**未**解决旧 Round 条件采样偏差，不能称为已经重新训练完整网络。

## 预算与检查点

在运行前固定验证调用预算 5%、10%、20%，并要求平均总 token 严格低于同题强模型独立回答。每个配方、每种分支、每个预算分别在可行点中按以下规则选择：

1. 验证集最终答对题数最高。
2. 同分选总 token 较低的点。
3. 再同分选调用率较低的点；跨 epoch 完全相同保留最早一轮。

阈值来自分数的无标签分位点：0--20% 每 1% 一个点，另报告 30%、50%、100%。同分样本整组处理，不能通过真实标签打破概率相同的并列。保存的是逐题概率阈值，而非依赖整批排序的部署 top-k。**验证调用预算不能保证分布变化后实际部署调用率不超过同一数值。**

每个点都通过实际 Gate -> 原 Round / 固定 R1 的动作读取已保存候选，不能使用 Oracle 轮次代理。训练缺失的反思不补造，不报告完整训练集的端到端准确率。

## 运行命令

```bash
cd /path/to/adaptive_reflection
THREADS=4 RUN_DIR=outputs/math_gate_cpu_ablation_new \
  ./scripts/run_math_gate_cpu_ablation.sh
```

已有 `RUN_DIR` 不覆盖，重复运行请用新目录。默认输入为 `outputs/fixed_math_policy_cpu_20260915_151132_seed42`；`SOURCE_RUN` 可指定其他兼容来源，但必须与同一缓存及固定划分完整对齐。缓存会检查输入文本、ID 顺序、指纹、分片范围和有限值；出错即停止，不静默重编码。

## 输出

- `protocol.json`：运行前规则、源数据/检查点哈希、缓存指纹、配方和预算；accuracy_tolerance 为 null。
- `baselines.json`：Direct、固定 R1/R2、强模型独立回答、候选 Oracle、未修改的原策略。
- 每组 `history.json`：每轮 loss、正类 precision/recall/F1、ROC-AUC、AP 和各阈值实际答案/成本结果。
- 每组 `epoch_XX.png`、`training_curve.png`：每轮累计曲线。
- 每组 `selected_points.json`、`gate_<branch>_<budget>.pkl`：每个预算的开发候选，非默认部署模型。
- `reference_round.pkl`：原 Round 的只读副本，仅供 original_round 分支使用。
- `summary.json` / `REPORT.md`：全部组别的比较，包括精度差、输入/输出/总 token、调用率、修复和引错。
- `status.json`：运行状态；全部导出检查点必须经缓存逐题重放一致后才标记 completed。

## 解释限制

仍只有 5976 条可用 Gate 训练题、185 条反思正例和 1492 条内部验证题。验证题参与多个 epoch、配方及阈值选择，收益可能偏乐观；不能直接据此报告独立测试显著性或非劣结论。5 道原始生成异常保留在审计里。跨模型 token 不等于 FLOPs/费用，部署中本地编码器与头的耗时还需另外测量。

## 冻结策略后的逐题错误归因

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python -m src.analyze_math_policy_errors \
  --ablation outputs/math_gate_cpu_ablation_20260916_seed42 \
  --output outputs/math_gate_error_analysis_new \
  --threads 4
```

此入口只分析已保存的原 Round 分支策略，以及最初未改阈值的策略，不重新训练或搜索阈值。复用原向量，并核对数据指纹、逐题预测和成本。输出目录不能覆盖已有分析。

归因先把 Direct 错误且至少有一个正确反思的题分成：已修复、Gate 未触发、Gate 已触发但 Round 选错。Direct 正确而最终错误的题另列为引错，避免重复计数。两个反思都错误的题仅标记为“现有候选未修复”，不算 Round 误选。

第二轮对比同题保存的第一轮，分为带来修复、两轮均对、两轮均错、引入错误；后面三种均记录额外 token。还保留 Gate 漏检题在当前 Round 下能否被修复的反事实标记。局部 Oracle 仅用于定位瓶颈，不属于部署方案，也不保证调用预算。

`REPORT.md` 和 `summary.json` 汇总统计；每个策略子目录保存全部 `decisions.jsonl` / CSV 及六类筛选清单，`examples.jsonl` 提供按 ID 查阅的原始问题和候选回答。每条决策保留预测、概率、正确性与 token，便于人工检查；**这些内部验证错误题不回流训练集**。

## Gate 排序目标对照

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python -m src.math_gate_ranking \
  --reference outputs/math_gate_cpu_ablation_20260916_seed42 \
  --output outputs/math_gate_ranking_new \
  --threads 4
```

只比较两种目标，不搜索更多排序权重：

- `weighted_ce_control`：上一轮加权小 MLP 的交叉熵。
- `weighted_ce_pairwise`：相同交叉熵加 0.5 倍 pairwise logistic 排序项。

令 `s = logit_reflect - logit_direct`，排序项为同一训练批次中全部正负对的 `mean softplus(s_negative - s_positive)`。正类仍是 `best_action != direct`，没有将“Direct 错误但所有反思也错”改标为正类；单类别批次排序项为零。它是带监督的排序学习，不是无监督学习或强化学习。

两组均使用 `4096 -> 16 -> 2` 的 ReLU/Dropout(0.2) 头、AdamW、学习率 0.0003、weight decay 0.01、seed 42、batch size 128、10 epoch，保留原始类别分布，不额外过采样。编码器、Round、数据划分、预算和阈值选择规则不变。

训练会核对两组初始化权重和全部批次 ID 顺序的哈希一致，并要求 CE 对照精确复现上一轮加权小 MLP 的六个预算结果。只读缓存，不重新编码；不读取上一阶段错误清单或独立测试集。

输出包括逐轮曲线、`objective_curve.png`、`LOSS_COMPARISON.md` 和 `controlled_comparison_audit.json`。主曲线中的 loss 是未加权交叉熵，history.json 另外记录加权 CE、全配对排序诊断项、组合目标，以及训练批次排序项均值；这些量不能混为同一条 loss。

本轮同样不设精度容差、不自动选论文主模型。无论排序结果改善或下降，都保留完整对照，不根据验证表现继续追加参数搜索。

## 三种子排序损失复核

固定两种损失及所有超参数，仅将 Gate 的训练随机种子扩展为 7、21、42。原 Round 一直固定，不能将这组标准差称为完整 Gate/Round 系统重训的随机性。保留先前 seed 42 原始输出，只补跑其余两个种子：

```bash
cd /path/to/adaptive_reflection
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python -m src.math_gate_ranking \
  --seed 7 --output outputs/math_gate_ranking_20260916_seed7 --threads 4
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python -m src.math_gate_ranking \
  --seed 21 --output outputs/math_gate_ranking_20260916_seed21 --threads 4
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python -m src.summarize_math_gate_multiseed \
  --runs outputs/math_gate_ranking_20260916_seed7 \
         outputs/math_gate_ranking_20260916_seed21 \
         outputs/math_gate_ranking_20260916_seed42 \
  --output outputs/math_gate_ranking_multiseed_20260916 --threads 4
```

这些目录已完成后不得覆盖；复跑请给新目录。训练 CLI 默认仍为 seed 42；仅该种子检查与旧 seed 42 的六点精确复现，其他种子不要求得到相同数值。每个种子内部两种损失必须具有相同初始化和批次顺序。

汇总先核对完整配方网格、相同协议/数据哈希/原 Round，再用缓存重新重放全部 36 个模型。`manifest.json` 保留旧 seed 42 与新运行各自代码哈希。`REPORT.md` 给出两种分支、三个预算的均值和样本标准差（ddof=1）、同种子排序减 CE 的配对差值、全部逐 seed 指标与所选 epoch。原始输入/输出/总 token 均保留，不只统计输出。

不按结果剔除种子，不选择最佳 seed，不将重复使用的 1492 个问题扩充为 4476 个独立样本。三种子统计仍是内部开发结果，不解除此前验证反馈或五个未完成生成样本的限制。

## 相对 Direct 的提升与随机 Gate 基线

研究目标调整为：用少量反思改善基础模型，向强模型质量靠近，并报告真实质量/成本权衡；不要求追平强模型，不设置精度容差。Direct 是主要有效性参照，固定反思是效率参照，强模型独立回答是质量参照。本轮不更改既有策略、检查点、预算或阈值。

```bash
cd /path/to/adaptive_reflection
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python -m src.analyze_gate_random_baseline \
  --source outputs/math_gate_ranking_multiseed_20260916 \
  --output outputs/math_gate_random_baseline_20260916 --threads 4
```

输出目录必须不存在；已完成的分析不要覆盖。共分析两种损失、两种分支、三个预算、三个训练种子的 36 个冻结检查点，不挑选最好的一次。

- `uniform_gate`：每次在全部题中无放回均匀抽 K 题，K 等于对应策略**实际**调用数，再用同一冻结 Round 选择反思轮次。
- `round_stratified_gate`：在 Round 预测为 R1/R2 的题中分别随机抽 K1/K2 题，与策略实际轮次调用数也一致。这是基于现有冻结 Round 输出的对照，不按标签或成本分组。
- fixed_r1 分支触发后只用一轮，两种随机抽样的分布相同。
- 每个点模拟 10,000 次，无放回抽样、不按正确性选择随机种子。汇总主表使用随机方法的解析期望，另外保存模拟均值、标准差及分位带。

精确 McNemar 只在每个种子的二值答案正确性上计算，比较策略与 Direct。36 个策略和 3 个固定参照共 39 次检验使用同一 Holm 校正族。对三个种子先在同题平均差值，再按题进行 10,000 次 paired bootstrap，报告相对 Direct、随机 Gate 解析期望、同轮次数随机解析期望的准确率与总 token 差区间。区间固定已有检查点和随机调用概率，不重新进行模型选择；不把未中心化 bootstrap 尾部比例当作假设检验 p 值。

相同调用数不等于相同 token：被选题目长度、响应长度和轮次构成都可能影响成本。输出同时保存输入、输出与总 token，不把这个实验称为严格等 token 对照。随机重复分位带与 bootstrap 区间分别表示随机策略抽样波动和当前题目经验分布的不确定性，不能混用。

结果在 `REPORT.md`、`summary.json`、`per_policy.json`。`decisions.jsonl` 保存全部策略的逐题动作及评分；`random_draws.npz` 保存模拟结果，每列含义见 summary.json 的 draw_columns。整个流程核对源文件/模型哈希并重放检查点，不训练、不加载编码器、不访问 LLM 或独立测试集。

**所有显著性统计仅用于探索性诊断。** 此前已在同一验证集挑选过 epoch、阈值及配方，即使 Holm 后 p 很小，也不构成独立确认实验，更不能因此抹去历史验证反馈。
