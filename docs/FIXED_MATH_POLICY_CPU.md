# 固定划分上的 Gate/Round CPU 训练

本入口只读取已保存的数学候选，不启动 vLLM、不调用 Base/Reflector、不读取 GSM8K test，也不继续 QA。CPU 加载的是冻结的 Qwen3-0.6B-Base 编码器，不是生成模型。

## 数据边界

复用 `outputs/math_full_training_v1/manifest.json` 中的划分，不再使用随机 `train_test_split`。

| 用途 | 原定题数 | 当前可用 | 监督分布 |
| --- | ---: | ---: | --- |
| Gate 训练 | 5978 | 5976 | direct 5791 / 需要反思 185 |
| Round 训练 | 仅实际完整候选 | 359 | reflect_1 319 / reflect_2 40 |
| 两个头共用内部验证 | 1495 | 1492 | direct 1453 / reflect_1 32 / reflect_2 7 |

5 道未解决生成异常保留在原失败账本及 `split_audit.json`，不修改原生成完成状态。这次是临时可用样本上的训练，不能报告为完整测试集结论。1492 题全部有三种动作和强模型独立回答，支持一致的离线比较；训练中 Direct 正确题没有反思观测，不能计算全训练集的最终答案准确率或错误引入率。

始终预测 Direct 的训练动作分类准确率已达 96.90%，因此分类准确率不是主要选择标准。Round 训练集只覆盖具有完整候选的 Direct 错题，存在条件采样偏差；不能将这些题上的表现解释为全部部署题的表现。

## 模型与选择规则

- 输入仅为 `Question` 和 `Initial answer` 文本；不输入金答案、正确性、难度、verifier 或 token 数。
- 编码器全部冻结，以 FP32 在 CPU 前向；读取最后两层，各层 masked mean 与最后有效 token 拼接为 4096 维。
- 默认策略输入上限 2048 token，不做额外字符裁剪。这批输入最长 1165 token，原先 1024 会裁剪 3 题，因此在训练前提高上限，保留全部当前输入。逐题长度和被裁剪 ID 保存在缓存的 `input_length_audit.json`；后续换数据也须复核。这是编码器输入限制，不是生成模型输出截断；不修改任何已有回答。
- Gate 和 Round 各有独立的训练侧 StandardScaler 和 `4096 -> 64 -> 16 -> 2` ReLU MLP。Gate 不加类别权重；Round 用训练候选的逆类别频率加权。
- AdamW，学习率 0.001，weight decay 0.001，batch size 128，默认 seed 42、10 epoch。seed 只影响头初始化和训练顺序，不改变题目划分。
- Round 在反思前用初始文本选 R1/R2，使用 argmax；Gate 通过固定的 0--1、步长 0.05 网格（另含始终反思端点）校准 Direct 概率阈值。
- 每轮使用同一轮两个头的**真实联合预测**，按内部验证最终正确题数优先、相同则总 token 较低、再相同则最早 epoch 选检查点。不使用最优反思作为 Gate 的乐观代理。
- 验证集参与 epoch 和阈值选择，最终需要锁定模型后在独立测试集评价；不能据此声称泛化已被证实。

## 运行

```bash
cd /path/to/adaptive_reflection
THREADS=4 RUN_DIR=outputs/fixed_math_policy_cpu_seed42 \
  ./scripts/run_fixed_math_policy_cpu.sh
```

脚本强制 CPU，屏蔽 CUDA，并禁止 Hugging Face 联网下载。默认最多用 4 个 PyTorch CPU 线程。向量按 32 题保存一个原子缓存分片，训练与验证共享原始向量，但标准化严格只在各自训练题上拟合。缓存指纹覆盖模型文件、输入文本、ID 顺序、层选择、截断上限和提取代码版本。

提取中断可用同一命令续跑。训练已经产生 `history.json` 后请使用新的 `RUN_DIR`，避免覆盖已有实验；同配置原始向量会复用。改变 seed 不改变向量指纹，可复用缓存。脚本不使用 GPU，也不会终止其他人的任务。

## 输出与读法

- `training_protocol.json` / `split_audit.json`：启动前固定规则及数据范围。
- `status.json`：当前编码、训练、完成或失败状态。
- `gate.pkl` / `round.pkl`：同一最佳 epoch 的两个头，包含 scaler、编码器配置和校准阈值。
- `history.json`：每轮两个头的训练/验证 loss、正类 precision/recall/F1、联合策略结果，以及未校准阈值 0.5 的对照。
- `training_curve.png` / `epoch_XX.png`：累计曲线和各轮保存版本，包括 loss、最终答案准确率、总 token、修复/引错数。
- `baselines.json`：Direct、固定 R1/R2、强模型独立回答和准确率优先 Oracle。
- `selected_validation.json`：所选 epoch、完整决策列表、输入/输出/总 token、反思率、修复/引错率、奖励及缺失题覆盖范围。

token 按已保存路径累计输入和输出；R1/R2 包含 Base 及实际反思调用，强模型独立回答不加 Base 成本。本地策略没有生成式 LLM token，但 CPU 前向耗时不为零，不能把 token 降低直接解释为部署加速。跨模型 token 也不是统一 FLOPs/价格单位。

优先检查是否接近强模型独立回答的答案准确率，同时降低总成本。Oracle 只说明候选空间潜力；即使动作分类分数很高，若修复不足或引错增多，也不能认定策略有效。
