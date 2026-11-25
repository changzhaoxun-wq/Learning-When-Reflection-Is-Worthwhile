# 中文标题

> 工作稿，非投稿定稿。2026-09-18 补充 GSM8K 全量、包含执行失败的测试结果及事后误差归因。当前证据索引为 `results_audit_20260918/`，保留 `method_audit_20260917/` 作为历史快照。由 Codex 核对，尚待作者人工复核。不是无截断确认实验；未完成的迁移、解耦和部署开销实验仍留空。

学习何时值得反思：基于冻结表征的分层反思路由方法

# English Title

Learning When Reflection Is Worthwhile: Hierarchical Routing with Frozen Representations

## 中文摘要

固定轮次反思为不同样本分配相同的测试时计算预算，而现有自适应方法多依赖不确定性、过程评分或生成过程中的内部状态控制反思，这类代理信号并不直接表征额外反思能否真正改善答案。本文提出一种基于冻结文本表征的分层反思效用路由方法，将反思决策建模为问题—初始回答状态下的计算效用预测，在生成额外反思前联合决定是否反思及反思轮数。具体而言，问题刻画任务本身的求解需求，初始回答反映模型在该任务上的实际求解状态；本文利用冻结小型语言模型的高层隐藏状态对二者进行联合编码，并通过掩码均值池化与末位有效 token 池化分别聚合全局上下文和回答终止状态，形成用于反思路由的状态表示。通过离线比较直接回答、一轮及两轮反思的正确性与累计 token，以“正确性优先、成本次优”构造反思效用监督，分别训练 Gate 和 Round 进行分层计算分配。整个过程中生成模型与编码器均保持冻结，仅优化轻量分类头。在 GSM8K 全部 1319 个样本上，本方法三随机种子平均准确率达到 94.57%–95.30%，高于直接回答的 91.73%和固定一轮反思的 93.33%，同时平均总 token 减少 23.8%–58.1%。随机调用对照进一步表明，主要收益来自对高反思效用样本的有效识别，而非简单增加反思次数。固定反思虽然能够提高部分数学题的答案质量，却会对已经正确的问题产生不必要的生成成本，甚至引入新错误。本文将反思调用建模为一个测试时计算分配问题：基础模型先生成初始答案，冻结的小型编码器将问题与初始答案映射为多层隐藏表示，随后由分层策略选择器决定是否调用反思器以及执行一轮还是两轮反思。策略仅使用部署时可获得的文本表示，不依赖人工难度标签、标准答案或候选真实得分；训练监督由离线候选的正确性和输入输出 token 成本构造。在包含失败成本的 GSM8K 全量测试中，策略准确率达到 94.57%--95.30%，相较 Direct 的 93.33% 提升 1.24--1.97 个百分点，同时将平均总 token 控制在 406.67--738.46，低于固定一轮反思的 969.72。匹配调用数随机对照表明，策略学习到了部分反思价值较高的样本，但跨数据集收益具有明显任务依赖性：GSM1K 上存在有限迁移，SVAMP 上未稳定复现，MultiArith 上主要体现为低调用率下维持饱和准确率。[claim:C001] [evidence:E001,E003,E005,E009,E019,E020,E025,E02

## English Abstract

Fixed reflection can improve some mathematical answers, but it also spends unnecessary generation budget on already-correct instances and may introduce new errors. We formulate reflection as a test-time computation allocation problem. A base model first produces an initial answer; a frozen small encoder maps the question and that answer to multi-layer representations; and a hierarchical policy decides whether to invoke a reflector and whether to perform one or two reflection rounds. The policy uses only text available at deployment and does not rely on handcrafted difficulty labels, gold answers, or oracle candidate scores. It is trained from the correctness and input-output token costs of offline candidates. On the full GSM8K test set with failed attempts included in the accounting, the policy achieves 94.57--95.30% accuracy, improving over Direct at 93.33% by 1.24--1.97 percentage points, while using 406.67--738.46 total tokens per question compared with 969.72 for fixed one-round reflection. Matched-call random controls indicate that the policy learns to allocate reflection to some higher-value instances. Transfer gains are task-dependent: they are limited on GSM1K, do not consistently appear on SVAMP, and primarily take the form of low-cost accuracy preservation on saturated MultiArith.[claim:C002] [evidence:E001,E003,E005,E009,E019,E020,E025,E026]

## 关键词 / Keywords

数学推理；选择性反思；分层策略；冻结表征；质量与成本权衡 / Mathematical reasoning; selective reflection; hierarchical policy; frozen representations; quality-cost trade-offs

## 1. 引言 / Introduction

### 1.1 研究背景与动机 / Background and Motivation

### 1.2 研究问题与目标 / Research Questions and Objectives

### 1.3 主要贡献 / Contributions

## 2. 相关工作 / Related Work

### 2.1 数学推理与测试时计算 / Mathematical Reasoning and Test-Time Computation

### 2.2 反思与答案修订 / Reflection and Answer Revision

### 2.3 模型路由与选择性计算 / Model Routing and Selective Computation

### 2.4 反思策略学习与本文定位 / Reflection Policy Learning and Positioning

## 3. 方法 / Method

本节描述当前数学实验的实际实现，不沿用历史 QA 的候选排序器、顶部层微调网络或早期 `verify_reflect` 动作。下文的“强模型”指所选反思器及其独立作答对照，不预设它在每道题上都优于基础模型。[claim:C003] [evidence:E001,E003,E005,E011]

### 3.1 问题定义与动作空间 / Problem Formulation and Action Space

给定数学问题 \(q\)，冻结基础模型 \(M_B\) 生成可见初始响应 \(y_0\)。选择器仅观察 \(s_0=(q,y_0)\)，输出动作 \(a\in\mathcal A=\{D,R_1,R_2\}\)：直接保留初始回答，执行一轮反思，或执行两轮反思。标准答案 \(g\) 只参与离线监督和评测；对动作 \(a\) 记录正确性 \(C_a\in\{0,1\}\) 与累计成本 \(T_a\)。GSM8K 使用统一抽取器：优先读取 `####` 尾部、Final answer/Answer is/Answer 标记和 `\boxed{...}`，否则使用响应中的最后一个数值；主评分路径抽取带符号、逗号、整数/小数形式的数值；代码中的底层 `normalize_math_answer` 另支持简单 LaTeX 分数，但 `candidate()` 实际调用 `extract_final_number`。无法抽取数值的输出视为错误；可转为数值的预测与标准答案按绝对误差不超过 1e-6 判等；百分号不执行额外的百分比换算，也不使用词级 F1。研究目标是在有限额外调用下改善基础模型答案质量，并报告与固定反思、强模型独立回答的差距，不预设非劣容差。[claim:C004] [evidence:E001,E003,E007,E011]

### 3.2 框架概览 / Framework Overview

离线阶段包括候选生成、逐路径评分、监督标签构造、冻结文本编码、分类头训练及验证侧检查点选择。在线阶段先取得初始回答，随后用相同编码方式计算表示，Gate 决定保留回答还是调用反思；仅在选择反思时，Round 的预测决定执行一轮或两轮。策略在产生任何反思响应之前完成动作选择，没有调用后的候选采纳头，也不会读取实际反思结果后再次决定是否追加第二轮。缓存轨迹上的离线测试只是重放这一决策过程，不向选择器暴露其余候选。[claim:C005] [evidence:E003,E004,E005,E010,E012]

#### 图 1. 整体框架 / Figure 1. Overall Framework

<!-- 待按本节重绘：分开离线监督与在线调用；gold/候选评分只连训练侧，强模型独立回答仅作基线，不画成第四个部署动作。 -->

### 3.3 初始回答与候选反思生成 / Initial Answer and Reflection Candidate Generation

当前生成组合为本地别名 `Qwen3-8B-Instruct` 的基础模型与 `Qwen3-14B` 反思器，均不更新参数。基础模型显式关闭 thinking，反思器显式开启 thinking；两者温度为 0、生成 seed 为 42。这是固定角色配置下的协作比较，不是相同 thinking 行为或相同计算量的模型能力比较。独立强模型对照只接收问题和 Direct 模板，不接收基础回答。Direct 模板只要求简洁推导和 `Final answer: <number>`；反思模板额外要求核对所求量、题目条件和算术，保留正确答案、修正错误，并禁止最终数字行包含单位、Markdown 或 LaTeX。[claim:C006] [evidence:E001,E002]

生成依赖关系为：

\[
y_0=M_B(P_D(q)),\qquad
y_1=M_R(P_R(q,y_0,1)),\qquad
y_2=M_R(P_R(q,y_1,2)).
\]

这里的 \(y_r\) 是剥离 thinking 段后的可见响应，原始输出仍保存在调用日志中。第二轮接收第一轮可见修订，而不是额外拼接完整隐藏推理历史。一轮路径与两轮路径共享同一初始调用和同一第一轮候选，因此第二轮对比是在既有第一轮基础上的增量生成，不是重新独立采样一次第一轮。[claim:C007] [evidence:E002,E010,E011]

### 3.4 离线监督构造与正确性优先标签 / Offline Supervision and Correctness-First Labels

对候选完整的问题，最优动作按正确性优先、成本次优的字典序定义为：

\[
a^*=\arg\max_{a\in\mathcal A}(C_a,-T_a),\qquad
b^*=\mathbf 1[a^*
e D],\qquad
r^*=\arg\max_{a\in\{R_1,R_2\}}(C_a,-T_a).
\]

正确性和成本都相同时，实现保留动作枚举中较早的动作，顺序为 \(D,R_1,R_2\)。Gate 的正类是“存在更优反思动作”，不是单纯“Direct 答错”：若所有候选均错误，Direct 因成本更低仍为负类。Round 学习最优反思轮次，不以 Gate 正类为唯一训练条件；两轮均错误时也保留按成本产生的标签。平均奖励 \(C_a-10^{-4}T_a\) 仅用于报告，不用于本版标签或梯度目标。[claim:C008] [evidence:E001,E002,E003]

训练侧采用选择性候选构造：对所有问题生成 Direct，只对 Direct 错误的问题补充两轮反思。Direct 已正确时，任一反思都需要额外正成本，故在上述标签规则下可直接标为 \(D\)，无需伪造未观测反思。验证侧为每题生成完整候选；全量测试也按此完整协议执行。缺失反事实不能设为错误或零成本，选择性训练轨迹不能用于估计固定反思的完整训练准确率或对初始正确题的引错率。[claim:C009] [evidence:E003,E010,E011]

当前固定清单的计划训练/验证规模分别为 5978/1495 题；可用记录为 5976/1492 题。这里的减少不是“record-validity checks”后静默删除，而是由生成阶段明确登记的未解决输出预算异常造成：训练集缺失 `gsm8k-main-train-3328`（Base Direct 三次尝试均以 `finish_reason=length` 结束）和 `gsm8k-main-train-5280`（Qwen3-14B 的 `reflect_2` 三次尝试均以 `finish_reason=length` 结束）；验证集缺失 `gsm8k-main-train-1855`（Base Direct 同样耗尽输出预算）、`gsm8k-main-train-2720`（`reflect_2` 耗尽输出预算）和 `gsm8k-main-train-3562`（`reflect_1` 耗尽输出预算）。五个失败动作都经过三次递增输出预算尝试，最终触发 `output_budget_exhausted`，并被写入 failure ledger；Base 失败对应的 Qwen3-14B 记录则标记为 `base_output_unresolved` 而没有重复调用。

这些题没有被重标为错误，也没有被赋予零 token 成本。`3328` 和 `1855` 没有可用的完整 Direct 结果；`5280` 虽保留了 Direct 和 R1 的调用日志，但缺少 R2，因此没有形成完整的固定候选记录；`2720` 和 `3562` 也分别缺少 R2/R1。由于策略训练与共同候选比较只读取最终可用记录，五题不进入 5976/1492 的训练或验证分母，但原始输出、已付 token 和失败原因均保留在 `outputs/math_full_training_v1/failures/`、`coverage_audit.json` 和 `GENERATION_REVIEW_20260915.md` 中。主 Gate 使用全部 5976 个可用训练问题，其动作标签为 5791 个 Direct、145 个 R1、40 个 R2，即 185 个反思正例，约 3.10%。为公平比较层级结构，离线 flat Direct/R1/R2 三分类控制也使用全部 5976 个可逻辑标注的训练问题：Direct 正确的 5791 题直接标为 Direct，不虚构未生成的反思；Direct 错误的 185 题使用已保存候选中的正确性优先、成本次优标签，因此得到 5791/145/40 的三类分布。R1-repairability 控制同样覆盖 5976 题：Direct 正确样本是已知负例，Direct 错误样本按已观测的 R1 正确性标注，当前产生 5831 个负例和 145 个正例。只有 Round 的 R1/R2 轮次标签需要完整三动作候选，因此仍使用 359 道完整初始错误题，其中 319 个 R1、40 个 R2。旧版仅用 359 条训练 flat 控制的结果已被新离线报告 `outputs/teacher_control_ablation_layers_v2_fair/REPORT.md` superseded，不参与本文结论；这些控制均复用固定轨迹，未重新调用大模型。

### 3.5 冻结编码器与多层隐藏表示 / Frozen Encoder and Multi-Layer Hidden Representations

策略输入按 `Question:\n{question}\nInitial answer:\n{initial_answer}` 连接，不附加置信度、verifier 分数、人工难度、初始抽取答案或 token 成本字段。编码器为 `Qwen3-0.6B-Base`，以评估模式、FP32 和禁用梯度的方式运行。读取返回的最后两层隐藏状态 \(H^{(\ell)}\in\mathbb R^{L\times1024}\)，令 \(m_i\) 为非 padding 掩码、\(i^*\) 为最后一个有效 token 位置，构造：

\[
\mu^{(\ell)}=\frac{\sum_i m_iH_i^{(\ell)}}{\sum_i m_i},\qquad
v^{(\ell)}=H_{i^*}^{(\ell)},\qquad
z=[\mu^{(-2)};v^{(-2)};\mu^{(-1)};v^{(-1)}]\in\mathbb R^{4096}.
\]

**全部编码器参数冻结，最后两层仅被读取，不接受微调。**表征维度来自层数、隐藏宽度与两种池化的拼接，而不是由训练样本数量决定。每个分类头分别使用仅在自身训练子集上拟合的 StandardScaler，再接收标准化向量。向量维度没有预设的错误或难度含义；能否预测反思收益由离线监督检验，不能仅凭高维表示声称学到了可解释的错误机制。[claim:C011] [evidence:E003,E004,E005,E006,E008]

策略编码上限固定为 2048 token，缓存记录问题 ID、输入文本哈希、编码器文件和编码设置。当前实际进入编码器的 5976 个训练样本、1492 个验证样本和 1319 个测试样本的最大 tokenized 长度分别为 1165、1035 和 677，三部分均有 0 个裁剪样本，即 no evaluated encoder input was truncated。训练/验证仍有 5 个未解决的上游生成异常样本未进入编码评估，因此不将该结论扩展为所有计划中的 GSM8K 行。该上限不同于生成模型的上下文窗口，不能把生成结束正常等同于选择器看到了全部初始响应。测试向量改由 GPU 以 FP32 提取；与已完成的 CPU 前缀比较及数值偏差处理见附录 E。[claim:C012] [evidence:E003,E008,E012,E019,E021]

### 3.6 分层策略选择器 / Hierarchical Policy Selector

#### 3.6.1 是否反思：Gate / Whether to Reflect: Gate

当前受控对照中的 Gate 为 `Linear(4096,16) → ReLU → Dropout(0.2) → Linear(16,2)`。Softmax 输出 \(p_D,p_R\)，分别对应训练标签 Direct 和需要反思。部署规则为 \(p_D\ge\tau\) 时保留 Direct，否则进入反思分支；相等时保留 Direct。增大 \(\tau\) 会扩大潜在反思集合。这些是分类分数，不能直接视为校准后的真实正确概率。两种训练目标使用相同结构、初始化及批次顺序进行对照。[claim:C013] [evidence:E004,E005,E012]

#### 3.6.2 反思轮次：Round / How Many Rounds: Round

Round 为 `4096 → 64 → 16 → 2` 的 ReLU MLP，不使用 Gate 的 Dropout 层；输出分别对应 R1、R2，部署采用 argmax，相同分数时选择 R1。其输入仍是 \((q,y_0)\)，不是第一轮反思。Round 在早期固定划分训练中与原 Gate 成对选择检查点，本次保留 epoch 1 的 Round，后续 Gate 对照不再更新它。另设 `fixed_r1` 分支：沿用同一批 Gate 训练，只把反思动作固定为 R1，并分别在验证侧选阈值和 epoch，检验轮次选择的影响。[claim:C014] [evidence:E003,E004,E008,E015]

#### 图 2. 策略选择器网络 / Figure 2. Policy Selector Network

<!-- 待按本节重绘：单个全冻结编码器、最后两层 mean/last 拼接、各自训练拟合的 scaler、不同宽度的 Gate 与 Round。不要画成顶部 Transformer 层可训练，也不要画 QA Ranker。 -->

### 3.7 训练目标与辅助排序损失 / Training Objectives and Auxiliary Ranking Loss

令训练标签 \(b_i^*\in\{0,1\}\)，Gate 的加权交叉熵在批次 \(\mathcal B\) 上为：

\[
\mathcal L_{\mathrm{CE}}=-\frac{\sum_{i\in\mathcal B}w_{b_i^*}\log p(b_i^*\mid z_i)}{\sum_{i\in\mathcal B}w_{b_i^*}},\qquad
w_0=1,\quad w_1=\min\{5,\sqrt{n_0/n_1}\}.
\]

当前训练计数使正类权重取上限 5。辅助排序分数为 \(s_i=\ell_{i,R}-\ell_{i,D}\)，其中 \(\ell\) 为未归一化 logits；只在同一训练批次中构造正负对：

\[
\mathcal L_{\mathrm{rank}}=\frac{1}{|\mathcal B_+||\mathcal B_-|}
\sum_{i\in\mathcal B_+}\sum_{j\in\mathcal B_-}\log(1+\exp(s_j-s_i)),\qquad
\mathcal L_G=\mathcal L_{\mathrm{CE}}+\alpha\mathcal L_{\mathrm{rank}}.
\]

单类别批次的排序项为零。CE 对照取 \(\alpha=0\)，排序辅助版本取 \(\alpha=0.5\)；不进行正类重采样，也不将验证错误或验证配对回流训练。排序损失促使更优反思标签获得更高分数，但不直接优化最终答案准确率，也不保证泛化改善。Round 使用其训练子集上的逆类别频率加权交叉熵，权重为 \(n/(2n_k)\)，不加排序项。整个过程是离线监督学习，而不是在线强化学习。[claim:C015] [evidence:E003,E004,E005]

当前 Gate 以 AdamW、学习率 \(3\times10^{-4}\)、weight decay 0.01、batch size 128 训练 10 个 epoch，种子为 7、21、42。固定 Round 来源训练使用学习率 \(10^{-3}\)、weight decay \(10^{-3}\)、相同 batch size，最多 10 个 epoch。每轮保存损失、分类指标及选择后的验证答案质量与成本；展示的普通 CE 曲线是未加权诊断量，不等同于加权 CE 加排序的优化目标。三种子结果主要衡量 Gate 训练随机性：Round、编码器、划分和生成候选均固定，不是三次完整框架独立重训。[claim:C016] [evidence:E003,E005,E008,E009,E015]

### 3.8 检查点选择与调用预算校准 / Checkpoint Selection and Call-Budget Calibration

选择准则基于最终答案而非动作标签准确率。对每个 Gate epoch，从验证 \(p_D\) 的经验分位生成阈值：目标比例网格为 0 至 0.20、步长 0.01，另含 0.30、0.50、1.00；相同分数作为整体处理。对 \(b\in\{0.05,0.10,0.20\}\)，保留满足以下条件的候选组合：

\[
\widehat{\rho}_{\mathrm{val}}\le b,\qquad
\overline T_{\mathrm{val}}<\overline T_{\mathrm{strong\ independent,val}}.
\]

在可行组合中先最大化真实 Gate--Round 路径的答对题数，再最小化平均 token 和调用率；同 epoch 完全相同时取更小阈值，跨 epoch 同值保留最早检查点。该规则同时用于原 Round 与固定 R1 分支。阈值网格仅由分数产生，但选择 epoch 和阈值仍使用验证答案及候选成本，因此验证结果是开发结果。两个损失、两个分支、三个预算和三个种子共形成 36 个锁定策略，未根据测试结果指定“最佳主模型”。[claim:C017] [evidence:E004,E005,E009,E011]

验证预算只是校准约束，并非测试时的硬调用率上限。测试直接应用锁定阈值，不重新按测试分位截取前若干题，也不保证平均 token 仍低于强模型。当前协议未指定相对强模型的准确率容差，因此“达到强模型水平”或“保持准确率”的统计性结论均不能从预算可行性推出。[claim:C018] [evidence:E004,E011,E012]

### 3.9 部署输入、决策流程与信息边界 / Deployment Inputs, Decisions, and Information Boundaries

给定新问题，先生成 \(y_0\) 并编码为 \(z\)，经各头保存的标准化器得到分数。若 Gate 保留 Direct，则直接抽取 \(y_0\) 的答案；否则采用 Round 的 argmax 决定轮数，并在冻结反思器上依次执行相应次数。返回所选路径最后一次可见回答，不从未选择的候选中利用真实正确性挑答案。部署没有标准答案、LLM verifier 或手工状态特征，也没有强模型独立回答这一第四动作。独立强模型仅用于质量与成本参照；\(D/R_1/R_2\) 的有限候选 Oracle 使用真实标签作回溯选择，不是可部署策略。[claim:C019] [evidence:E003,E004,E011,E012]

### 3.10 输入输出成本核算 / Input and Output Cost Accounting

设逻辑调用 \(k\) 实际执行的尝试集合为 \(\mathcal J_k\)，定义：

\[
c_k=\sum_{j\in\mathcal J_k}(t^{\mathrm{in}}_{kj}+t^{\mathrm{out}}_{kj}),\quad
T_D=c_0,\quad T_{R_1}=c_0+c_1,\quad T_{R_2}=c_0+c_1+c_2.
\]

每次尝试读取服务端真实 usage，要求输入加输出等于 total；thinking 输出也属于输出成本。反思重复输入的问题与当前回答再次计费，初始调用在每条路径内只计一次。强模型独立回答基线仅计自身调用，不附加基础模型成本。离线一次性生成所有候选的构造总开销与部署所选路径成本分开；缓存重放不会把逻辑调用成本变成零，也不会把所有未执行候选的成本归给在线策略。[claim:C020] [evidence:E001,E002,E011,E012]

生成上下文为 32768 token，按 `prompt_tokens + 16384 + 512 ≤ 32768` 预检。输出请求预算依次为 4096、8192、16384；遇到 `finish_reason=length` 时用相同提示重新生成，而不是接着截断文本续写，所有尝试成本累计。自然结束后才提取可见响应；未闭合 thinking 段会触发错误。达到最大预算仍未完成的调用保留日志并进入异常清单，不冒充完整语义错误或被静默删除。本轮在独立的事后核算目录中完成包含 4 次执行失败的全量评估；失败路径计零作答得分并保留完整成本，不能宣称已经得到“无截断的全量测试结果”。[claim:C021] [evidence:E001,E002,E011,E020]

token 只是生成成本代理：不同规模模型每 token 的计算代价不同，冻结编码器和分类头也有本地推理开销。本文尚不能把 token 减少解释为已测得的同等比例延迟、FLOPs、能耗或费用下降。后续需分别报告输入、输出、总 token 以及策略前向延迟；当前方法不包含在线 verifier，其成本不能虚构为已测量。[claim:C022] [evidence:E001,E003,E012]

## 4. 实验设置 / Experimental Setup

### 4.1 数据集与固定划分 / Datasets and Fixed Splits

#### 表 1. 数据集与划分 / Table 1. Datasets and Splits

| 数据集 / Dataset        | 训练 / Train | 验证 / Validation | 测试 / Test | 用途 / Purpose          |
| ----------------------- | ------------ | ----------------- | ----------- | ----------------------- |
| GSM8K 计划问题清单      | 5978         | 1495              | 1319        | 固定问题级划分          |
| GSM8K 训练/验证可用记录 | 5976         | 1492              | 不适用      | 保留另外 2/3 题异常审计 |
| GSM8K 含失败测试核算    | 不适用       | 不适用            | 1319        | 所有方法保留完整分母    |

训练与验证使用既定划分中的可用记录，未在测试阶段补训。测试并非删除四道异常题，而是记录每个动作的成功或执行失败结果。两种覆盖口径不可混写；本轮不包含 QA。[claim:C024] [evidence:E008,E014,E020]

### 4.2 模型、提示词与生成协议 / Models, Prompts, and Generation Protocol

采用第 3.3、3.10 节的冻结生成协议。独立强模型与反思使用同一 Qwen3-14B，前者只接收问题，后者接收问题与待修订响应。角色间 thinking 设置不同，不能将差异仅归因于模型参数规模。[claim:C025] [evidence:E001,E002,E011]

### 4.3 输出完整性、重试与失败处理 / Output Completeness, Retries, and Failure Handling

原流水线保存 1315 个完全完成的测试行，另外 4 题分别在一次强模型独立回答或三次第二轮反思中耗尽预算。第二轮失败导致三次无依赖的独立回答被跳过，本轮只补齐这三次调用，不重跑其他候选。共核算 5276 个逻辑调用、5302 次付费尝试，输入/输出分别为 940084/2265102 token，合计 3205186 token。这是整个候选库的构造开销，不是某一种策略部署全部问题的成本。[claim:C026] [evidence:E020]

耗尽预算的动作保留空预测和执行失败标记，在系统准确率中计零分；部分推理不作为完整回答评分。所有实际尝试成本累计。该规则是查看失败后的事后补充，原“无未解决截断才发布”的入口仍保持阻塞，不将新增核算伪装成原协议自然完成。[claim:C027] [evidence:E020,E024]

### 4.4 策略训练与超参数 / Policy Training and Hyperparameters

权重、epoch 和阈值在本轮测试生成前锁定。报告两个训练目标、两个反思分支、三个验证预算及三个 Gate 种子的全部组合，不据测试结果选种子或重新训练；具体 epoch 和阈值见附录 D。[claim:C028] [evidence:E009,E019]

### 4.5 对照方法与随机基线 / Comparison Methods and Random Baselines

对照包括 Direct、固定一轮/两轮反思、强模型独立回答及有限候选 Oracle。随机 Gate 匹配每个策略的实际反思题数；同轮次随机对照进一步在固定 Round 预测的轮次分层中匹配 R1/R2 调用数。固定 R1 分支下两种随机对照相同。这些对照不匹配实际 token，不能写成等 token 优势。[claim:C029] [evidence:E017,E019]

### 4.6 评价指标与有限候选 Oracle / Metrics and Finite-Candidate Oracle

报告答案准确率、平均输入/输出/总 token、调用率、修复数、引错数和执行失败数。修复率分母为 88 道 Direct 错误题，引错率分母为 1231 道 Direct 正确题；总体净增益为修复减引错后除以 1319。Oracle 只在 Direct/R1/R2 中按真实正确性优先、成本次优选择，不含独立强模型，不可部署。[claim:C030] [evidence:E019,E022]

### 4.7 多种子实验与配对统计 / Multi-Seed Experiments and Paired Statistics

均值和样本标准差描述固定候选、固定 Round 下三个 Gate 种子的差异。组间比较先按题平均三种子得分，再按问题进行 10000 次配对 Bootstrap；随机对照另保留 10000 次调用模拟。每个单种子策略及三个非 Oracle 基线分别与 Direct 做双侧精确 McNemar 检验，39 项 p 值共同用 Holm 校正。表 6 的 p 范围是三个单种子检验，不是一个“种子平均 McNemar”。Bootstrap 区间未作多重比较校正；统计检验不能消除历史测试反馈。[claim:C031] [evidence:E017,E019,E024]

### 4.8 测试前锁定与复现协议 / Pre-Test Locking and Reproducibility Protocol

本轮不利用测试标签修改 prompt、权重、epoch 或阈值。补跑、执行失败计分和 CPU/GPU 数值复核单独记录。官方测试集曾在项目历史中被观察，本轮是锁定配置下、含事后核算的探索性测试，不是从未使用过的确认集。[claim:C032] [evidence:E011,E019,E020,E021]

## 5. 实验结果与分析 / Results and Analysis

### 5.1 数学主任务结果 / Main Mathematical Reasoning Results

#### 表 2. 主实验结果 / Table 2. Main Results

<!-- RESULTS:main:START -->

| 方法                  | 验证预算 | Acc. (%)      | 输入 token | 输出 token | 总 token        | 调用率 (%) |
| --------------------- | -------- | ------------- | ---------- | ---------- | --------------- | ---------- |
| direct                | --       | 93.33         | 97.31      | 157.68     | 254.98          | 0.00       |
| fixed_reflect_1       | --       | 95.75         | 429.72     | 540.00     | 969.72          | 100.00     |
| fixed_reflect_2       | --       | 95.83         | 618.68     | 1099.38    | 1718.06         | 100.00     |
| independent           | --       | 96.51         | 94.04      | 617.91     | 711.95          | 不适用     |
| oracle_accuracy_first | --       | 96.66         | 112.24     | 211.58     | 323.82          | 3.34       |
| CE/Gate-Round         | 5%       | 94.57 ± 0.04 | 169.43     | 330.91     | 500.34 ± 15.01 | 13.87      |
| CE/Gate-Round         | 10%      | 95.07 ± 0.20 | 202.26     | 393.30     | 595.55 ± 13.66 | 20.32      |
| CE/Gate-Round         | 20%      | 95.20 ± 0.04 | 261.96     | 476.50     | 738.46 ± 18.20 | 33.23      |
| CE/Gate-R1            | 5%       | 94.57 ± 0.12 | 153.49     | 253.76     | 407.24 ± 5.53  | 13.87      |
| CE/Gate-R1            | 10%      | 95.02 ± 0.23 | 178.34     | 281.67     | 460.01 ± 12.87 | 20.32      |
| CE/Gate-R1            | 20%      | 95.17 ± 0.12 | 218.06     | 322.13     | 540.19 ± 38.81 | 31.54      |
| CE+Rank/Gate-Round    | 5%       | 94.59 ± 0.22 | 171.11     | 347.73     | 518.84 ± 24.51 | 14.10      |
| CE+Rank/Gate-Round    | 10%      | 94.97 ± 0.16 | 201.60     | 392.12     | 593.73 ± 33.30 | 20.29      |
| CE+Rank/Gate-Round    | 20%      | 95.30 ± 0.08 | 261.32     | 474.52     | 735.85 ± 2.06  | 33.21      |
| CE+Rank/Gate-R1       | 5%       | 94.57 ± 0.24 | 153.78     | 252.89     | 406.67 ± 15.72 | 14.10      |
| CE+Rank/Gate-R1       | 10%      | 94.95 ± 0.19 | 177.62     | 278.78     | 456.40 ± 24.46 | 20.29      |
| CE+Rank/Gate-R1       | 20%      | 95.25 ± 0.04 | 219.29     | 324.94     | 544.24 ± 11.20 | 31.84      |

<!-- RESULTS:main:END -->

表 2 报告全部配置；Acc. 为百分比，± 为三个 Gate 种子的样本标准差。CE 为加权交叉熵，CE+Rank 加入辅助排序；Gate-Round 学习轮次，Gate-R1 固定一轮。5%/10%/20% 是验证预算标签，不是测试调用率；独立强模型的反思率不适用。[claim:C033] [evidence:E019,E022]

Direct 为 93.33%，固定 R1/R2 为 95.75%/95.83%，独立强模型为 96.51%。策略组均值为 94.57%–95.30%，相对 Direct 提升 1.24–1.97 个百分点，但未达到固定 R1 或强模型的点估计。总 token 为 406.67–738.46，体现的是相对固定反思减少调用的质量—成本权衡，不是同时优于所有基线。[claim:C034] [evidence:E019]

### 5.2 相对基础模型的修复与错误引入 / Repairs and Error Introduction Relative to the Base Model

#### 表 3. 错误转移 / Table 3. Error Transitions

<!-- RESULTS:transitions:START -->

| 方法                  | 修复数 | 引错数 | 修复率 (%) | 引错率 (%) | 净变化 (pp) | 执行失败数 |
| --------------------- | ------ | ------ | ---------- | ---------- | ----------- | ---------- |
| fixed_reflect_1       | 37     | 5      | 42.05      | 0.41       | +2.43       | 0          |
| fixed_reflect_2       | 44     | 11     | 50.00      | 0.89       | +2.50       | 3          |
| independent           | 58     | 16     | 65.91      | 1.30       | +3.18       | 1          |
| oracle_accuracy_first | 44     | 0      | 50.00      | 0.00       | +3.34       | 0          |

<!-- RESULTS:transitions:END -->

固定 R1 修复 37 题、引错 5 题，净增 32 题。第二轮相对第一轮额外修复 7 题、引错 6 题，净增仅 1 题，却平均增加 748.34 token。引错包含执行失败造成的零得分，不全是完整回答的语义退化。独立强模型的“修复/引错”只表示相对 Base 的正确性转移，不表示它接收了初始回答。[claim:C035] [evidence:E019,E020,E022]

### 5.3 学习选题与随机调用对照 / Learned Selection versus Random Calls

#### 表 4. 匹配调用数量的随机对照 / Table 4. Matched-Call Random Comparisons

<!-- RESULTS:random:START -->

| 策略               | 验证预算 | 策略 Acc. (%) | 均匀随机 Acc. (%) | 同轮次随机 Acc. (%) | 策略/均匀/同轮次 token |
| ------------------ | -------- | ------------- | ----------------- | ------------------- | ---------------------- |
| CE/Gate-Round      | 5%       | 94.57         | 93.64             | 93.65               | 500.34/400.64/409.63   |
| CE/Gate-Round      | 10%      | 95.07         | 93.79             | 93.81               | 595.55/468.30/487.89   |
| CE/Gate-Round      | 20%      | 95.20         | 94.08             | 94.11               | 738.46/603.87/640.05   |
| CE/Gate-R1         | 5%       | 94.57         | 93.66             | 93.66               | 407.24/354.15/354.15   |
| CE/Gate-R1         | 10%      | 95.02         | 93.82             | 93.82               | 460.01/400.21/400.21   |
| CE/Gate-R1         | 20%      | 95.17         | 94.09             | 94.09               | 540.19/480.40/480.40   |
| CE+Rank/Gate-Round | 5%       | 94.59         | 93.65             | 93.66               | 518.84/403.03/415.78   |
| CE+Rank/Gate-Round | 10%      | 94.97         | 93.79             | 93.81               | 593.73/468.03/488.57   |
| CE+Rank/Gate-Round | 20%      | 95.30         | 94.08             | 94.11               | 735.85/603.61/642.89   |
| CE+Rank/Gate-R1    | 5%       | 94.57         | 93.67             | 93.67               | 406.67/355.77/355.77   |
| CE+Rank/Gate-R1    | 10%      | 94.95         | 93.82             | 93.82               | 456.40/400.03/400.03   |
| CE+Rank/Gate-R1    | 20%      | 95.25         | 94.10             | 94.10               | 544.24/482.57/482.57   |

<!-- RESULTS:random:END -->

所有组准确率均高于两种匹配调用数量的随机期望，现有未作多重比较校正的配对区间下界均大于零；学习策略也比对应随机调用花费更多 token。因此结果支持学习选题，但不足以证明严格相同计算预算下的优势。完整区间和随机模拟保存在源 JSON 中。[claim:C036] [evidence:E019]

### 5.4 质量与成本权衡 / Quality-Cost Trade-offs

#### 图 3. 质量与成本曲线 / Figure 3. Quality-Cost Curves

![全部十二组策略的质量与成本，以及验证预算与实际测试调用率；误差线为三个 Gate 种子的标准差。](results_audit_20260918/quality_cost_test.png)

图 3 展示全部十二组配置；误差线为种子标准差而非置信区间，连线只连接离散预算点。准确率坐标未从零起始，完整数值见表 2。原 Round 在 20% 验证预算下测试调用率约 33.2%，平均 735.85–738.46 token 已高于强模型独立回答的 711.95。验证约束没有自动迁移到测试；固定 R1 更省，也不能因此事后将其包装成预先指定的主模型。[claim:C037] [evidence:E019,E023]

### 5.5 轮次分支与训练目标对照 / Round-Branch and Training-Objective Comparisons

为直接检验“分层拆分是否优于 flat 三分类”，我们补充了公平的离线 teacher-control 对照。该实验复用相同候选轨迹、Last-1 至 Last-6 表征、固定划分和三个训练种子，不产生新的 LLM 调用。Flat classifier 使用全部 5976 个可逻辑标注的训练行：Direct 正确样本在正确性优先、成本次优规则下必然标为 Direct；Direct 错误样本使用已保存的 D/R1/R2 候选构造最优动作标签。因而其训练标签为 5791/145/40，而不是旧版不公平的 359 行。R1-repairability 控制也使用全部 5976 行，将 Direct 正确样本作为确定的负例；Round 仍必须使用 359 个完整三动作行，因为 R1 与 R2 的相对优劣无法由 Direct 正确性推断。

公平 flat 三分类在 Last-1 至 Last-6 上的测试准确率为 93.40%–93.68%，平均总 token 为 269.8–297.0，反思率为 1.04%–2.81%。该 flat 控制沿用不加权的三分类交叉熵，其反思率较低主要由 5791/145/40 的标签不平衡造成，因此不能把它与经过反思预算校准的 Gate-Round 直接当作等成本比较。R1-repairability 控制的测试准确率为 94.36%–94.49%。这些结果修正了旧版仅用 359 个训练样本的 flat 对照，但并不单独证明层级结构在所有预算下都优于 flat 分类器；完整协议和逐层数字见 `outputs/teacher_control_ablation_layers_v2_fair/REPORT.md`。

| k | Flat Acc. | Flat Token | Flat Reflect. |
|---:|---:|---:|---:|
| 1 | 93.40% ± 0.08 | 271.7 ± 11.1 | 1.04% |
| 2 | 93.43% ± 0.04 | 270.1 ± 8.2 | 1.29% |
| 3 | 93.53% ± 0.16 | 279.2 ± 11.3 | 1.72% |
| 4 | 93.43% ± 0.04 | 269.8 ± 7.3 | 1.34% |
| 5 | 93.53% ± 0.04 | 280.6 ± 7.9 | 2.05% |
| 6 | 93.68% ± 0.04 | 297.0 ± 8.5 | 2.81% |

#### 表 5. Oracle 差距归因 / Table 5. Oracle-Gap Attribution

完整目标/分支对照已列于表 2，不另挑最好配置。以 20% 验证预算的 CE+Rank 为例，Gate-Round 为 95.30% 与 735.85 token，Gate-R1 为 95.25% 与 544.24 token；前者高 0.05 个百分点，却多用约 191.61 token。两分支分别校准 epoch/阈值，这不是对同一已选题集合只改变轮数的纯因果消融。排序辅助也未在所有预算下优于 CE，例如 10% 档两个分支均略低，不能声称有普遍显著改善。[claim:C038] [evidence:E019]

表 5 改列与有限候选 Oracle 的事后差距归因；每格为三个种子的平均题数，定义见第 5.7 节。

<!-- RESULTS:attribution:START -->

| 策略               | 验证预算 | Gate 漏修复 | 反思轮次漏修复 | 对初始正确题引错 | 执行失败 (重叠计数) |
| ------------------ | -------- | ----------- | -------------- | ---------------- | ------------------- |
| CE/Gate-Round      | 5%       | 21.33       | 4.67           | 1.67             | 1.33                |
| CE/Gate-Round      | 10%      | 14.67       | 4.33           | 2.00             | 2.00                |
| CE/Gate-Round      | 20%      | 10.33       | 5.00           | 4.00             | 2.00                |
| CE/Gate-R1         | 5%       | 21.33       | 5.33           | 1.00             | 0.00                |
| CE/Gate-R1         | 10%      | 14.67       | 5.67           | 1.33             | 0.00                |
| CE/Gate-R1         | 20%      | 10.00       | 7.00           | 2.67             | 0.00                |
| CE+Rank/Gate-Round | 5%       | 20.67       | 4.33           | 2.33             | 2.00                |
| CE+Rank/Gate-Round | 10%      | 15.67       | 4.67           | 2.00             | 2.00                |
| CE+Rank/Gate-Round | 20%      | 10.00       | 5.00           | 3.00             | 2.00                |
| CE+Rank/Gate-R1    | 5%       | 20.67       | 5.33           | 1.67             | 0.00                |
| CE+Rank/Gate-R1    | 10%      | 15.67       | 5.67           | 1.33             | 0.00                |
| CE+Rank/Gate-R1    | 20%      | 9.67        | 6.67           | 2.33             | 0.00                |

<!-- RESULTS:attribution:END -->

### 5.6 多种子稳定性与配对差异 / Multi-Seed Stability and Paired Differences

#### 表 6. 配对统计 / Table 6. Paired Statistics

<!-- RESULTS:statistics:START -->

| 策略               | 验证预算 | 相对 Direct ΔAcc. [95% CI] (pp) | 三种子 Holm p 范围 | 相对强模型 ΔAcc. [95% CI] (pp) |
| ------------------ | -------- | -------------------------------- | ------------------ | ------------------------------- |
| CE/Gate-Round      | 5%       | +1.24 [+0.63, +1.90]             | 0.0013–0.00165    | -1.95 [-3.03, -0.86]            |
| CE/Gate-Round      | 10%      | +1.74 [+1.04, +2.50]             | 2.74e-05–0.000968 | -1.44 [-2.48, -0.40]            |
| CE/Gate-Round      | 20%      | +1.87 [+1.06, +2.70]             | 0.00013–0.000386  | -1.31 [-2.32, -0.30]            |
| CE/Gate-R1         | 5%       | +1.24 [+0.66, +1.87]             | 0.000574–0.00165  | -1.95 [-3.03, -0.86]            |
| CE/Gate-R1         | 10%      | +1.69 [+1.01, +2.43]             | 1.54e-05–0.000859 | -1.49 [-2.53, -0.45]            |
| CE/Gate-R1         | 20%      | +1.84 [+1.09, +2.65]             | 8.43e-06–0.000386 | -1.34 [-2.35, -0.33]            |
| CE+Rank/Gate-Round | 5%       | +1.26 [+0.63, +1.95]             | 0.000968–0.00219  | -1.92 [-2.98, -0.86]            |
| CE+Rank/Gate-Round | 10%      | +1.64 [+0.96, +2.38]             | 8.79e-05–0.000574 | -1.54 [-2.55, -0.51]            |
| CE+Rank/Gate-Round | 20%      | +1.97 [+1.16, +2.81]             | 1.62e-05–0.000142 | -1.21 [-2.22, -0.18]            |
| CE+Rank/Gate-R1    | 5%       | +1.24 [+0.63, +1.90]             | 0.000574–0.00219  | -1.95 [-3.01, -0.88]            |
| CE+Rank/Gate-R1    | 10%      | +1.62 [+0.96, +2.35]             | 2.74e-05–0.000859 | -1.57 [-2.58, -0.53]            |
| CE+Rank/Gate-R1    | 20%      | +1.92 [+1.16, +2.73]             | 8.43e-06–0.00013  | -1.26 [-2.27, -0.25]            |

<!-- RESULTS:statistics:END -->

全部单种子策略相对 Direct 的 McNemar 检验经 39 项 Holm 校正后仍有 p<0.003；三种子平均的配对 Bootstrap 也支持正向差值。这是固定候选上的统计证据，不消除历史测试反馈。全部组相对强模型的准确率差区间均低于零，不能将“接近”解释为等效或非劣。原 Round、CE+Rank、20% 档相对固定 R1 的差值为 −0.45 pp，区间 [−1.01, +0.08] pp；包含零不证明保持准确率。完整原始 p、校正 p 和成本差区间保留在源 JSON 中。[claim:C039] [evidence:E019]

### 5.7 难度分组与 Gate/Round 误差归因 / Difficulty Groups and Gate/Round Error Attribution

新增事后、互斥的 Oracle 差距归因：①初始错误且有正确反思，却选择 Direct，记 Gate 漏修复；②初始错误且选择了反思，但选定轮次错误、另一轮正确，记轮次漏修复；③初始正确却选到错误或失败反思，记引错。三项之和等于 Oracle 正确数减策略正确数。执行失败是重叠标记，不能作为第四项再相加。Gate-R1 的第二项反映固定一轮的动作限制，不是它有一个 Round 网络却选错。[claim:C040] [evidence:E022]

CE+Rank、原 Round、20% 档平均 Gate 漏修复 10 题、轮次漏修复 5 题、引错 3 题，共解释相对 Oracle 的 18 题平均差距。该诊断不用于测试侧调参。question-only hard 分组在本版尚未独立核对，不填入历史不同协议的分桶结果。[claim:C041] [evidence:E022]

### 5.8 跨数学数据集迁移 / Transfer across Mathematical Datasets

迁移实验复用 GSM8K 验证集锁定的策略配置，不在目标数据集上重新选择阈值、epoch 或种子。候选生成、输入输出 token 统计和失败处理均采用同一份锁定协议；因此这些结果检验的是策略的跨数据集适应性，而不是重新训练后的最优上界。随机 Gate 使用与对应策略相同的反思调用数；同轮次随机 Gate 还匹配 R1/R2 的调用比例。所有随机值均为解析期望，配对区间以问题为单位计算。[claim:C049] [evidence:E025,E026]

#### 表 7. 跨数据集结果 / Table 7. Cross-Dataset Results

| 数据集 / Dataset | 方法 / Method            | 准确率 / Accuracy | 总 token / Total Tokens | 调用率 / Call Rate |
| ---------------- | ------------------------ | ----------------- | ----------------------- | ------------------ |
| GSM1K            | Direct                   | 90.21             | 328.0                   | 0.00               |
| GSM1K            | Fixed R1                 | 94.19             | 1202.0                  | 100.00             |
| GSM1K            | Locked policy            | 93.03             | 986.2                   | 32.95              |
| GSM1K            | Matched-call random Gate | 92.22             | 929.7                   | 32.95              |
| SVAMP            | Direct                   | 93.33             | 249.8                   | 0.00               |
| SVAMP            | Fixed R1                 | 95.00             | 904.4                   | 100.00             |
| SVAMP            | Locked policy            | 93.00             | 531.6                   | 12.33              |
| SVAMP            | Matched-call random Gate | 93.45             | 501.2                   | 12.33              |
| MultiArith       | Direct                   | 100.00            | 142.7                   | 0.00               |
| MultiArith       | Fixed R1                 | 99.44             | 572.7                   | 100.00             |
| MultiArith       | Locked policy            | 100.00            | 189.5                   | 7.22               |
| MultiArith       | Matched-call random Gate | 99.98             | 189.7                   | 7.22               |

表 7 只报告由 GSM8K 验证协议预先锁定的单一策略，不再按迁移测试集选择最高点。完整 36 点及其描述性排序见 `outputs/math_transfer_locked_v1/evaluation/summary_report_v2/summary_table.md`，预锁定策略的独立报告见 `outputs/math_transfer_locked_v1/evaluation/prelocked_policy_v1/REPORT.md`。固定策略相对同轮次匹配随机 Gate 在 GSM1K 上提高 0.81 个百分点，但在 SVAMP 上低 0.45 个百分点；MultiArith 已饱和，二者差异约为 0.02 个百分点。该结果说明迁移并非在所有数据集上成功，当前证据更支持 GSM1K 上的有限迁移，不能声称普遍跨数据集有效。由于迁移数据此前已被实验过程观察，这些结果仍应标为探索性证据，不应表述为独立确认。[claim:C050] [evidence:E025,E026]

质量—成本散点图见图 4。横轴为包含输入和输出的平均总 token，纵轴为准确率；灰色点为 Direct、固定反思和独立强模型，彩色点为不同锁定策略。图中 token 差异不等于 FLOPs、延迟或费用差异，策略选择器本地前向开销未计入该图。[claim:C051] [evidence:E025]

进一步的离线错误分析显示，GSM1K 上锁定策略修复 38 题、引入 4 个错误，并漏掉 17 个“至少一个反思候选正确”的 Direct 错误；SVAMP 上没有成功修复题目，且漏掉 6 个可修复错误，同时有 1 个反思引入错误。SVAMP 的 R1/R2 候选本身分别只修复 6/7 个 Direct 错误，说明其主要瓶颈首先是候选质量，而不是 Gate 单独造成的漏选。SVAMP 的问题和初始回答平均长度也短于 GSM1K，但这只是描述性差异，不能据此认定长度导致泛化失败。MultiArith 没有 Direct 错误，反思没有可提升空间。详细统计见 `outputs/math_transfer_locked_v1/evaluation/failure_analysis_v1/REPORT.md`。[claim:C052] [evidence:E027]

#### 图 4. 迁移数据集质量—成本 / Figure 4. Transfer Quality-Cost Trade-off

![迁移数据集上的质量—成本散点图。灰色点为 Direct、固定反思和独立强模型；彩色点为锁定策略。](../outputs/math_transfer_locked_v1/evaluation/summary_report_v2/transfer_quality_cost.png)

### 5.9 基础模型与反思器解耦 / Base Model and Reflector Decoupling

<!-- 待当前协议下完成。必须区分替换组件后重新训练策略与冻结策略直接迁移。 -->

### 5.10 策略选择器的部署开销 / Deployment Overhead of the Policy Selector

<!-- 待测；缓存向量上的头预测耗时不能替代在线编码器加分类头的端到端延迟。 -->

## 6. 讨论 / Discussion

### 6.1 候选质量与策略选择能力 / Candidate Quality and Policy Selection Capability

有限候选 Oracle 为 96.66%，而策略组均值上限为 95.30%，表明候选空间尚未充分利用。Oracle 不可部署，也不能作为继续优化当前测试标签的目标。后续应在训练/验证侧研究 Gate 漏修复和轮次风险，再冻结配置评估。[claim:C042] [evidence:E019,E022]

### 6.2 反思与额外独立求解的区别 / Reflection versus Additional Independent Solving

独立强模型准确率 96.51%、平均 711.95 token。部分策略少用 token 但准确率也较低；20% 档原 Round 平均更贵。尚无同预算延长 Direct、相同强模型路由或等 FLOPs 对照，不能把所有提升归因于反思机制本身。[claim:C043] [evidence:E019]

### 6.3 类别不平衡、过拟合与泛化 / Class Imbalance, Overfitting, and Generalization

训练正例稀缺，验证预算与测试实际调用率又有偏离；动作分类准确率不足以评价系统。应同时分析漏修复、引错、执行失败和成本，当前数字不能确定偏移原因，更不能证明扩大网络或增加 epoch 必然有效。[claim:C044] [evidence:E014,E019,E022]

### 6.4 适用范围与证据边界 / Scope and Evidence Boundaries

当前证据来自 GSM8K 及三个规模较小的数学迁移集，并使用一组固定模型角色配置。迁移结果支持学习选题相对随机调用的有限泛化，但学习轮次的额外收益仍有限；目标数据集规模和历史观察限制了确认性解释。部署端到端延迟及独立确认集仍待完成，不用旧 thinking/预算不一致的结果替代。[claim:C045] [evidence:E019,E022,E025,E026]

## 7. 结论 / Conclusion

本文实现冻结表征驱动的分层反思选择器。GSM8K 含执行失败的全量测试显示，锁定策略以额外计算提高基础模型准确率，并在匹配调用数量的随机对照中体现选题能力；相对固定反思则呈现质量与 token 权衡。跨数据集结果显示，这种选题优势在 GSM1K 上更明显，在 SVAMP 上较小，而在已饱和的 MultiArith 上主要体现为低调用率。轮次网络边际收益有限，策略未达到独立强模型准确率。所有失败和成本均保留，迁移结论仍属于探索性证据。[claim:C046] [evidence:E019,E020,E022,E025,E026]

## 局限性 / Limitations

本轮含事后失败核算与运行设备数值复核，历史测试集也曾参与开发反馈，不是纯确认性结论。训练/验证仍有异常，选择性 Round 训练缺少初始正确题监督。三个种子仅覆盖 Gate；统计区间不包含完整生成随机性，也不能消除多重探索。角色 thinking 设置不同，token 不等于跨模型计算费用，尚无本地编码加生成的端到端延迟。迁移、完整表征消融、同预算机制对照及新协议 hard 分析不能写成已完成。[claim:C047] [evidence:E008,E019,E020,E021,E022]

## 伦理声明 / Ethics Statement

## 参考文献 / References

<!-- 研究相关工作待逐项核验后补充；下项仅为本轮写作工具归属，不构成方法新颖性或实验有效性证据。 -->

Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner. 2026. *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065 . 本轮使用其中的 scientific-writing 流程辅助本地核对与起草；作者须复核正文，并按目标 venue 的政策决定最终披露方式。[claim:C023] [evidence:E016]

## 附录 A. 完整提示词与协议指纹 / Appendix A. Full Prompts and Protocol Fingerprints

## 附录 B. 网络结构、参数与超参数 / Appendix B. Architecture, Parameters, and Hyperparameters

## 附录 C. 训练曲线与检查点 / Appendix C. Training Curves and Checkpoints

## 附录 D. 全部种子与逐预算结果 / Appendix D. All Seeds and Per-Budget Results

保留全部 36 个锁定点及其验证选择的 epoch/阈值，不从测试结果筛选种子。[claim:C048] [evidence:E019]

<!-- RESULTS:seeds:START -->

| 策略               | 验证预算 | seed | epoch | 阈值 p_D   | Acc. (%) | 总 token | 调用率 (%) | 执行失败 |
| ------------------ | -------- | ---- | ----- | ---------- | -------- | -------- | ---------- | -------- |
| CE/Gate-Round      | 5%       | 7    | 2     | 0.63087815 | 94.62    | 516.83   | 14.33      | 2        |
| CE/Gate-Round      | 10%      | 7    | 9     | 0.82649493 | 95.15    | 579.94   | 19.18      | 2        |
| CE/Gate-Round      | 20%      | 7    | 9     | 0.92536324 | 95.15    | 739.86   | 33.28      | 2        |
| CE/Gate-R1         | 5%       | 7    | 2     | 0.63087815 | 94.69    | 410.17   | 14.33      | 0        |
| CE/Gate-R1         | 10%      | 7    | 9     | 0.82649493 | 95.07    | 445.25   | 19.18      | 0        |
| CE/Gate-R1         | 20%      | 7    | 9     | 0.92536324 | 95.07    | 550.35   | 33.28      | 0        |
| CE+Rank/Gate-Round | 5%       | 7    | 3     | 0.60429364 | 94.84    | 543.74   | 16.00      | 2        |
| CE+Rank/Gate-Round | 10%      | 7    | 9     | 0.79316550 | 94.92    | 571.76   | 19.18      | 2        |
| CE+Rank/Gate-Round | 20%      | 7    | 9     | 0.91106761 | 95.30    | 736.69   | 33.28      | 2        |
| CE+Rank/Gate-R1    | 5%       | 7    | 3     | 0.60429364 | 94.84    | 422.38   | 16.00      | 0        |
| CE+Rank/Gate-R1    | 10%      | 7    | 9     | 0.79316550 | 94.92    | 442.71   | 19.18      | 0        |
| CE+Rank/Gate-R1    | 20%      | 7    | 9     | 0.91106761 | 95.22    | 554.27   | 33.28      | 0        |
| CE/Gate-Round      | 5%       | 21   | 7     | 0.68123317 | 94.54    | 487.45   | 13.27      | 1        |
| CE/Gate-Round      | 10%      | 21   | 10    | 0.85501999 | 94.84    | 601.43   | 20.77      | 2        |
| CE/Gate-Round      | 20%      | 21   | 5     | 0.86105746 | 95.22    | 755.92   | 35.10      | 2        |
| CE/Gate-R1         | 5%       | 21   | 7     | 0.68123317 | 94.54    | 400.87   | 13.27      | 0        |
| CE/Gate-R1         | 10%      | 21   | 10    | 0.85501999 | 94.77    | 465.85   | 20.77      | 0        |
| CE/Gate-R1         | 20%      | 21   | 5     | 0.86105746 | 95.15    | 572.90   | 35.10      | 0        |
| CE+Rank/Gate-Round | 5%       | 21   | 6     | 0.62779808 | 94.47    | 518.01   | 14.18      | 2        |
| CE+Rank/Gate-Round | 10%      | 21   | 8     | 0.75163519 | 94.84    | 577.38   | 18.65      | 2        |
| CE+Rank/Gate-Round | 20%      | 21   | 2     | 0.69057906 | 95.22    | 737.35   | 33.43      | 2        |
| CE+Rank/Gate-R1    | 5%       | 21   | 6     | 0.62779808 | 94.39    | 406.68   | 14.18      | 0        |
| CE+Rank/Gate-R1    | 10%      | 21   | 8     | 0.75163519 | 94.77    | 441.85   | 18.65      | 0        |
| CE+Rank/Gate-R1    | 20%      | 21   | 10    | 0.89865929 | 95.22    | 532.15   | 29.34      | 0        |
| CE/Gate-Round      | 5%       | 42   | 4     | 0.58537531 | 94.54    | 496.74   | 14.03      | 1        |
| CE/Gate-Round      | 10%      | 42   | 4     | 0.71807122 | 95.22    | 605.29   | 21.00      | 2        |
| CE/Gate-Round      | 20%      | 42   | 4     | 0.80509770 | 95.22    | 719.61   | 31.31      | 2        |
| CE/Gate-R1         | 5%       | 42   | 4     | 0.58537531 | 94.47    | 410.70   | 14.03      | 0        |
| CE/Gate-R1         | 10%      | 42   | 4     | 0.71807122 | 95.22    | 468.93   | 21.00      | 0        |
| CE/Gate-R1         | 20%      | 42   | 8     | 0.86050230 | 95.30    | 497.31   | 26.23      | 0        |
| CE+Rank/Gate-Round | 5%       | 42   | 8     | 0.57748824 | 94.47    | 494.75   | 12.13      | 2        |
| CE+Rank/Gate-Round | 10%      | 42   | 4     | 0.61163545 | 95.15    | 632.05   | 23.05      | 2        |
| CE+Rank/Gate-Round | 20%      | 42   | 8     | 0.84018952 | 95.38    | 733.50   | 32.90      | 2        |
| CE+Rank/Gate-R1    | 5%       | 42   | 8     | 0.57748824 | 94.47    | 390.94   | 12.13      | 0        |
| CE+Rank/Gate-R1    | 10%      | 42   | 4     | 0.61163545 | 95.15    | 484.65   | 23.05      | 0        |
| CE+Rank/Gate-R1    | 20%      | 42   | 8     | 0.84018952 | 95.30    | 546.29   | 32.90      | 0        |

<!-- RESULTS:seeds:END -->

## 附录 E. 输出完整性与成本审计 / Appendix E. Output Completeness and Cost Audits

执行失败为 `gsm8k-main-test-22` 的独立强模型回答及 `test-62`、`test-744`、`test-1082` 的第二轮反思。原始响应、结束原因、usage 和哈希均保留。并未恢复这些失败的语义答案，只是为全部测试题给出可核算的系统结果，没有删除截断题。[claim:C049] [evidence:E020]

测试向量采用 GPU FP32、禁用 TF32，分类和统计在 CPU。CPU 前缀共 384 题，1572864 个元素中有 2 个未通过严格逐元素容差，但 Round 和全部 36 个策略的动作均相同。原失败审计保留；另行事后、无答案标签的向量级复核接受运行调整，最大绝对差为 0.00036621，RMSE 约 0.00001197。此前缀检查不是全测试逐位一致的证明。GPU 编码约 21 秒只是批次工程测量，不是端到端单题延迟或完整加速比。[claim:C050] [evidence:E021,E026]

主来源为 `outputs/math_full_test_failure_accounted_v1/evaluation/summary.json`。表格、逐策略归因和图 3 可由 `python paper/summarize_completed_math_test.py --update-draft-tables` 重建，不加载模型、不调用 LLM、不更新策略。新增证据与主张记录位于 `results_audit_20260918/`，尚待作者人工复核。[claim:C051] [evidence:E022,E023]

## 附录 F. 典型修复与失败案例 / Appendix F. Representative Repairs and Failures
