# HotpotQA 独立开发集实验

## 为什么先做这一步

现有 7,378 条 HotpotQA 测试轨迹中，direct 的二值正确率为 0.6586，
双候选 oracle 为 0.6937。最近的策略实验没有得到稳定净收益：
last-three-block、10% 开发集调用预算的版本修复 7 题、破坏 4 题，
McNemar p=0.5488，同时 token 成本明显增加。utility ranker 则修复 9 题、
破坏 62 题。

原固定验证划分的 direct 正确率约 0.7914，官方测试轨迹约为 0.6586。
元数据核对显示，原训练集含 easy、medium 和 hard，官方 HotpotQA 验证集
全部为 hard，题型约为 80% bridge、20% comparison。这支持“分布差异影响
校准”的假设，但不能证明它是唯一原因；gate 在测试上的 AUC 约为 0.541，
说明排序能力本身也可能不足。

本轮先检查固定策略在新的 train-hard 数据上的修复和破坏行为，再决定是否
进行分来源校准。当前流程只负责准备和生成候选，尚未训练或重新校准策略。

## 固定数据划分

| 集合 | 数量 | Bridge | Comparison | 用途 |
| --- | ---: | ---: | ---: | --- |
| development | 2,000 | 1,600 | 400 | 分析与选择阈值 |
| confirmation | 2,000 | 1,600 | 400 | 阈值锁定后的一次性独立确认 |

- 数据来自官方 train 的 hard 子集，使用原 v13 的 distractor 数据源。
- 排除旧 v5/v7 训练轨迹和此前全部 8,000 条 direct 训练样本。
- 排除与官方测试集重合的原始 ID 或规范化题目。规范化包含 Unicode NFKC、
  大小写和空白；该检查不等价于语义级改写查重。
- 两个新集合之间也按 ID 和题目查重。固定随机种子为 `20260908`。
- 80/20 比例只依据题型元数据；抽样不依赖 gold、direct 正确性或候选收益。
- confirmation 仅预留样本，本脚本不对其调用模型、训练或评估。

输出目录：

```text
outputs/qa_direct_first/hotpotqa_hard_development_v1/
  manifest.json
  development.jsonl
  confirmation.jsonl
  configs/{direct,v5,v7}.json
  profiles/{v5,v7}.json
  direct.jsonl                  # 运行 direct 后产生
  v5_raw.jsonl, v7_raw.jsonl    # 每题都生成 reflect_1
  v5.jsonl, v7.jsonl            # 审计、F1 重标注后产生
  audits/
  generation_complete.json     # 生成与审计均完成后产生
```

`manifest.json` 保存数据、配置、模板及参考策略的哈希。准备阶段可以重复执行，
内容不同则报错；生成中途改变协议必须使用新目录。

## 生成协议

Base 使用 `Qwen3-8B-Instruct`，Reflector 使用
`DeepSeek-R1-Distill-Qwen-7B`。Base 的 direct prompt 保持
`qa_direct_first_v2`，关闭 thinking；温度为 0、seed 为 42。

v5 从历史训练调用中恢复模板，其两轮模板拼接的 SHA-256 必须等于原
v13 protocol。v7 与当前模板及保存的训练调用逐字核对。两套 profile
分别固定各自的 completion suffix 和格式修复模板，避免把 v7 的
`Decision:` 前缀加到 v5 输出上。当前默认 prompt 文件没有被替换。

每题只生成一次 Base 回答，v5/v7 共享这一份回答与 Base token 记录。
所有 development 题都生成两种反思候选，包括 direct 正确的题；否则无法
无偏地测量错误引入率，也不能可靠比较“调用后保留原答案”与“采用反思”。

固定窗口为 8192。Base 输出上限 1024，重试上限 2048；QA 反思主调用
上限为 256。主调用截断后执行已有的短答案恢复流程，未得到完整输出时
停止写入该条记录。脚本先检查所有 Base prompt；生成 Base 后，再用实际
initial answer 检查每个反思 profile 及格式修复、截断恢复的预算。

费用按逻辑调用累计全部输入和输出，包含失败重试、截断恢复及格式修复。
双候选选择器一旦调用反思，成本为 Base 一次加 v5/v7 两个反思增量，
即使最后保留 direct 也要计入。策略编码器的推理延迟/GPU 成本应单独报告，
不能把 LLM token 数等同于完整系统计算成本。

## 运行命令

```bash
cd /path/to/adaptive_reflection

# 固定划分和协议检查；不调用 LLM
PHASE=prepare ./scripts/run_qa_hard_development.sh
PHASE=check ./scripts/run_qa_hard_development.sh
PHASE=preflight ./scripts/run_qa_hard_development.sh

# Base 在 8000、Reflector 在 8001 服务就绪后，一键生成并审计
./scripts/run_qa_hard_development.sh
```

当前脚本复用已有模型服务，不自动抢占显卡或停止其他进程。
显存有限时可分阶段：先只运行 Base 服务并生成 direct，再切换到 Reflector。
完整 direct 已存在时，重新执行 `all` 会复用它，无需 Base 服务继续在线。

```bash
PHASE=direct ./scripts/run_qa_hard_development.sh
PHASE=v5 ./scripts/run_qa_hard_development.sh
PHASE=v7 ./scripts/run_qa_hard_development.sh
PHASE=audit ./scripts/run_qa_hard_development.sh
```

中断后重跑同一命令。程序核对已完成 ID、顺序、prompt profile 和 token，
续写剩余样本。`pipeline.lock` 防止同一目录被两个任务同时写入。

F1 重标注沿用既有 gold 审计规则，疑似过长 gold 会进入 quarantine。
这些排除会记录在审计文件中，应在论文中披露，不能声称过滤后的分数就是
完整官方测试分数。标准 EM、标准 token F1 与 `F1 >= 0.8` 二值正确率
分别报告；后者不是 HotpotQA 官方 EM。

## 生成完成后的决策

先检查 development 的 direct、v5、v7、dual oracle，以及各来源的 fixes、
harms、标准 EM/F1 和调用成本。然后在 development 上固定权重，使用
一个 gate 阈值和每个反思来源各自的 adoption margin。预先固定开发集
最大调用比例 10%、候选网格 21 点；依次以二值答案准确率、标准 F1 和
更低 token 成本选择。无正收益时保留 direct，并报告负结果。

阈值确定后再生成/评估预留 confirmation，做配对 bootstrap 和 McNemar。
已有官方测试结果曾用于多次方法探索，应作为探索性结果披露；新的确认集
才用于检查这轮改动是否泛化。开发集的提升本身不能作为最终论文效果。

更强编码器或新的训练目标留到这一步之后：如果开发集 oracle 的提升空间
仍小，优先改善候选质量；若 oracle 有明显收益而选择器无法分离 repairs
与 harms，再考虑策略表示或训练数据的改动。

## 2026-09-09 完成记录

开发集 2000 题的 Direct、v5、v7 已全部生成。沿用既有 gold 隔离规则排除
9 题后，1991 题进入对齐评估。排除不是按模型得分筛选，也不意味着已经人工
确认这些 gold 错误；本表不是完整官方基准结果。

两个反思来源均无未解决截断、缺失最终答案、缺失结束原因或 usage 不一致。
v5/v7 分别有 16/2 个逻辑调用发生触限后恢复，恢复成本已累计。因此不能
把这次审计描述为“从未截断”。本轮没有修改生成 prompt 或参考模型权重。

| 方法 | Acc@0.8 | 标准 EM | 标准 F1 | 平均输入 | 平均输出 | 总 Token | 调用率 | 修复/引错 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct | 0.6655 | 0.6062 | 0.7481 | 1308.9 | 7.8 | 1316.8 | 0.0000 | 0/0 |
| Fixed v5 | 0.3144 | 0.2752 | 0.4629 | 3182.4 | 67.1 | 3249.5 | 1.0000 | 61/760 |
| Fixed v7 | 0.6489 | 0.5912 | 0.7341 | 2933.8 | 19.9 | 2953.7 | 1.0000 | 8/41 |
| 冻结参考策略 | 0.6640 | 0.6052 | 0.7471 | 1990.6 | 22.5 | 2013.0 | 0.1813 | 2/5 |
| 分来源校准 | 0.6660 | 0.6067 | 0.7483 | 1506.0 | 12.1 | 1518.1 | 0.0502 | 1/0 |
| 双候选 Oracle | 0.6986 | 0.6344 | 0.7703 | 1423.9 | 10.1 | 1434.0 | 0.0331 | 66/0 |

Acc@0.8 是项目二值指标，不是官方 EM；Oracle 不可部署。双候选策略一旦
调用就计入两个来源，即使最后未采纳。分来源校准实际调用 100 题，采纳
18 题，只净修复 1 题；相对 Direct 增加约 15.3% token，奖励由 0.6206
降为 0.6007。开发集诊断 McNemar p=1.0，且这些题同时用于门限选择，
不能将微小正差当作独立显著性证据。QA 部署基线仍保留 Direct。

校准入口只读取开发集缓存，不调用 LLM，不更新编码器：

```bash
python -m src.calibrate_qa_hard_development \
  --root outputs/qa_direct_first/hotpotqa_hard_development_v1
```

已固定 21 点分位数网格，搜索 1936 个可行组合；最多允许调用 199 题。
锁定 Gate 阈值 0.5914868116378791，v5 margin -0.1685391664505005，
v7 margin 0.05468481779098511。每个来源先判断自己的 margin，再在通过的
候选中选择分数最高者；两者都未通过则保留 Direct。负 margin 是开发侧
校准结果，不表示对应分数已经概率校准。

结果位于本实验目录的 source_calibration/：
- policy_lock.json：不可静默覆盖的门限、权重、输入与校准代码哈希。
- development_report.json、REPORT.md：完整指标和开发诊断。
- development_decisions.jsonl：逐题调用与采纳来源。

确认集仍未生成候选或评估。下一步仅应用上述锁定规则做独立确认，不能在
确认集重扫门限；本地策略延迟仍需另测。完整集敏感性分析须保留隔离的 9 题。
校准与协议相关的 13 项单元测试已通过。

## 2026-09-10 独立确认已启动

以上“确认集仍未生成”是 2026-09-09 的状态。确认入口现已实现并启动，
使用同一预留 2000 题、同一 Base/Reflector 参数和 v5/v7 提示。
配置只替换数据路径并增加 held_out_confirmation 角色说明；不会修改旧
开发集或重新扫描阈值。两个数据集共用的审计函数新增可选确认 ID 参数，
默认开发集行为不变。确认运行前另存代码和协议哈希。

```bash
cd /path/to/adaptive_reflection
PHASE=preflight ./scripts/run_qa_hard_confirmation.sh
GPU=2 ./scripts/run_qa_hard_confirmation.sh
```

2026-09-10 00:42:12 已后台启动，PID 2240570；不要同时重复启动。
日志为 outputs/qa_direct_first/hotpotqa_hard_confirmation_v1/pipeline_20260910_004212.log。
全部 Base 题的窗口预检查通过，最大需求 4143/8192；反思预算在真实
Direct 生成后再逐题预检查。17 项协议、校准及确认测试通过。

默认输出目录：outputs/qa_direct_first/hotpotqa_hard_confirmation_v1/
- manifest.json、policy_lock.json：运行协议和开发侧锁定阈值，不可静默改写。
- direct.jsonl、v5_raw.jsonl、v7_raw.jsonl：全部原始候选。
- audits/、v5.jsonl、v7.jsonl：沿用既有隔离规则的审计及对齐数据。
- status.json：generation、locked_policy_inference、complete 或 failed。
- reference_decisions.jsonl、prediction_lock.json：小模型分数缓存和来源哈希。
- confirmation_report.json、confirmation_decisions.jsonl、REPORT.md：最终结果。
- evaluation_complete.json：只有全部评估完成才写入。

流程始终检查所有题的反思候选，不能只反思错误题来构建固定方法对照。
策略决策使用固定分数阈值，不读取候选真实正确性；真实标签仅用于统计及
不可部署的 Oracle。主比较为锁定策略减 Direct，二值指标使用双侧精确
McNemar，EM、F1、输入/输出/总 Token 与奖励给出 10,000 次问题级配对
Bootstrap 的 95% 区间（seed=20260910）。次要区间不作多重比较校正，
未显著不代表等效。实际确认调用率不强行截到 10%，因为该限制是开发侧
选择条件；不能按确认集分数重新选分位数。

当前是已启动而非已完成。启动日志、运行状态和失败记录保留；报错时不会
将失败题静默丢弃。中断后重跑上述命令，已完成候选和已验证预测缓存会复用。
生成接口复用现有 Base/Reflector 服务，GPU=2 只控制小策略模型推断。

## 2026-09-10 用户暂停 QA

按用户要求，确认流水线进程组 2240570 和本轮 QA 专用 vLLM 服务进程组
1071770 已停止；8000/8001 端口已释放，未停止其他任务。QA 工作暂不继续。
暂停时保存 Direct 2000 条、v5 1110 条，v7 尚未生成；所有已有 JSONL
均可读取。确认集没有完成评估，不能据此填写确认实验结果。
运行目录的 pause_record.json 保存暂停时间、文件哈希和上一阶段状态，
status.json 已标记 paused_by_user。没有删除数据、锁文件、模型或日志。
未来恢复需先启动同协议服务，再执行原确认命令；目前不要自动恢复。
