# 锁定策略后的 GSM8K 全量测试

## 本轮目标

用少量反思提升基础模型，报告相对 Direct 的答案质量变化，并与固定反思、匹配调用数量的随机策略及强模型独立回答比较。不要求追平强模型，不设置精度容差，不根据测试集选择新的阈值或最优种子。QA 不参与本轮。

## 测试前锁定

`outputs/math_full_test_v1/policy_lock.json` 在第一次测试生成前锁定：

- CE / CE+排序两个配方、原 Round / 固定 R1 两个分支、5% / 10% / 20% 三个验证预算和 7 / 21 / 42 三个训练种子，共 36 个已有检查点。
- 每个检查点的模型文件、SHA-256、epoch、阈值；原 Round 文件及哈希。
- 编码器、训练向量来源、训练/验证数据指纹及其缺失样本。
- 1319 道官方测试题清单及指纹；只检查与训练/验证的 ID 和规范化问题重叠，不以测试得分挑题。

模型按字节复制到 `locked_models/`，保留原模型。测试只应用已锁定阈值，不做测试端 top-k、不重新校准。实际调用率可以超出验证预算，必须如实报告。

原训练域仍有 2 道训练和 3 道验证生成异常；本轮不修补标签、不重训策略。这限制了训练数据完整性的表述，但不会把测试集缩小成同样的缺失集合。GSM8K 官方测试在项目早期被观察过，因此本轮是冻结新协议后的评测，不是完全未接触过的确认集。

## 生成协议

- Base：Qwen3-8B-Instruct 本地别名，显式 non-thinking，TP=1。
- Reflector：Qwen3-14B，thinking 开启，TP=2。
- 与 `outputs/math_full_training_v1/protocol.json` 逐项校验模型元数据、分片状态、提示词、thinking、评分代码与生成预算。
- 请求为 completions，使用本地 tokenizer 构建相同 chat template；temperature=0、seed=42。
- 上下文 32768；输出 4096 -> 8192 -> 16384，额外预留 512。窗口检查使用真实提示和上一轮可见回答，不截短输入。
- 每题均生成 Direct、R1、R2、独立 Qwen3-14B。R2 使用 R1 的可见回答；独立回答不读取 Base 或反思答案。
- 所有题都生成反思，包括 Direct 正确题，不能以标准答案控制测试候选覆盖。
- 重试保存到 `calls/`，输入+输出均计费。R1 路径包含 Base，R2 包含 Base+R1+R2；独立强模型路径不包含 Base。

仍触及最大预算时隔离为失败，不当作正常答错，不静默删除。其余题继续；任何未解决失败都会阻止最终评测及完整报告。网络错误、协议改变等异常直接停止，可在核查后续跑。已经付费的完整调用可复用，不因一行尚未提交就重复生成。

## 运行与恢复

```bash
cd /path/to/adaptive_reflection

# 仅预检和锁定，不启动 vLLM。
bash scripts/run_math_full_test.sh --preflight-only --base-gpu 0 --reflector-gpus 0,1

# 正式运行或在中断后继续同一个目录。
bash scripts/run_math_full_test.sh --base-gpu 0 --reflector-gpus 0,1
```

运行期间不要重复启动，`pipeline.lock` 防止并发修改。GPU 0 先运行 Base；Base 服务关闭后，再在 GPU 0、1 上启动 Qwen3-14B。只有入口自己创建的服务进程组会被关闭，不清理其他用户或项目的服务。若显存余量不足，按 60 秒间隔等待，不强行加载。服务使用 localhost:8014，端口被占用时拒绝复用。

启动脚本显式将 vLLM 环境的 `bin/` 和 `${CUDA_HOME:-/usr/local/cuda}/bin/` 加入 PATH，保证 FlashInfer 能找到已安装的 ninja 和 nvcc。2026-09-16 首次后台启动在编译采样器时因 PATH 缺失 ninja 而失败，尚无任何测试生成；原日志保留，环境修复和重新启动记录保存在 `startup_retries/`，模型和请求设置未变。

改变 GPU 放置等配置也会触发协议差异检查，不应绕过锁定文件强行续跑。本入口不继承旧测试模型或旧测试轨迹。

## 自动离线评测

只有 1319 题四种候选全部完整、逐调用审计通过后，才生成 `policy_test.jsonl`。随后自动关闭生成服务并在 CPU 上执行：

1. 使用冻结的 Qwen3-0.6B-Base 编码器提取测试特征，最后两层 mean+last 池化，4096 维；复用训练最大输入长度 2048。
2. 仅使用各训练模型已有 scaler，不在测试集拟合标准化器。编码器输入截断与生成截断分开记录；不根据测试内容扩大训练输入长度。
3. 应用全部 36 个 Gate 检查点和冻结 Round；报告最终正确率、调用率、输入/输出/总 token、修复/引错及三种子均值/标准差。
4. 报告 Direct、固定 R1/R2、强模型独立回答及有限候选 Oracle。
5. 补充实际调用数匹配、以及 R1/R2 数量也匹配的随机对照，每组 10,000 次；不选择最佳随机 seed。
6. 按题配对 Bootstrap 10,000 次，三个策略种子先在同题平均；另列每个二值策略的 McNemar 和 39 次比较的 Holm 校正。

相同调用数不等于相同 token；必须同时检查随机对照成本。独立强模型不是反思 Oracle 的候选。历史测试反馈与多比较限制仍需披露。

## 查看状态

```bash
cat outputs/math_full_test_v1/status.json
cat outputs/math_full_test_v1/launcher.json
```

后台启动的主日志路径记录在 `launcher.json`。`active_server.json` 记录当前/最近一次由本入口创建的模型服务。`stage_eta_seconds` 只预测当前生成阶段，不是整个生成加 CPU 评测的剩余时间。

最终输出：

- `completion_audit.json`：完整性、未解决失败、全部已付费调用 usage。
- `evaluation/REPORT.md`、`evaluation/summary.json`：固定策略测试报告。
- `evaluation/decisions.jsonl`：逐题动作及评分。
- `evaluation/random_draws.npz`：随机重复。
- `evaluation/encoder_audit.json`、`test_feature_cache/`：CPU 向量及策略输入长度审计。

只有根目录 `status.json` 为 `completed` 且 `test_metrics_released=true` 时，才表示生成与离线评测均结束。
