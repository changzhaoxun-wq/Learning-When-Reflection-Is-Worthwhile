# 当前协议的数学迁移实验

## 离线策略评测（2026-09-21）

完整轨迹生成后，运行以下命令，不调用 Base/Reflector 服务：

```bash
python -u -m src.evaluate_math_transfer --gpu 0
```

小编码器在 GPU 上采用 FP32、SDPA math、关闭 TF32，读取最后两层的 mean+last 向量；
沿用训练时 2048 token 输入长度、纯文本输入和已训练的标准化器，裁剪情况单独记录。
分类头与统计计算在 CPU 上完成。该入口不训练、不搜索阈值、不指定测试最优主模型。

首先从本地调用日志重构全部结果，核对得分与费用；缺失响应直接停止，不能重新调用模型补齐。
每套数据的完整集、问题去重集和非重合去重集均报告全部 36 个冻结策略。
三种子均值与标准差描述训练随机性；配对 Bootstrap 按规范化问题聚类重采样 10000 次。
McNemar 仅用于无重复问题子集，并在各子集内对 36 策略和 3 个固定对照作 39 项 Holm 校正；
没有做跨数据集/子集的总体多重校正，Bootstrap 区间不是同时置信区间。

`evaluation/REPORT.md` 给出完整集主表；`evaluation/<dataset>/` 保存各预定子集 JSON、
全部策略动作、概率及向量缓存审计；`evaluation/summary.json` 是完整汇总。
`evaluation/status.json` 的 completed 才表示本轮离线评测结束；根目录旧 status 是生成阶段记录。

GSM1K 的 `gsm1k-test-277` 曾有独立强模型调用丢失响应及 usage。重试值已保存，
但原尝试费用仍未知，故包含该题的独立强模型 token 仅为已知下界、reward 为上界。
未知值不能补零。该异常没有已知的 Base/R1/R2 用量丢失，不改变策略选择或答案评分。
本轮未补随机匹配调用数对照、在线延迟或新训练消融，不能宣称这些实验已完成。

## 单卡续跑修订（2026-09-19）

用户批准强模型改在 GPU 0 单卡执行，TP=1、显存比例 0.80；不再等待 GPU 2。
Base 1685 题全部保留并逐项验证哈希。原始运行协议保留不覆盖，
`single_gpu_amendment.json` 记录旧/新布局、实现哈希和所有受保护文件。
尚无旧双卡强模型调用，因此本轮强模型候选全部使用同一单卡配置。
模型、prompt、thinking、上下文 32768 和输出上限均未改动。
TP 变化可能造成数值差异，不能声称与 GSM8K 双卡输出逐位等价。

本目录此后只能用以下入口续跑，不要再用下文的历史双卡命令：

```bash
bash scripts/resume_math_transfer_single_gpu.sh --gpu 0
```

添加 `--preflight-only` 只核验；已有后台任务时不要重复启动。
单卡启动失败会保留日志并停止，不自动降低输出预算。
完成时另存 `single_gpu_export_provenance.json`，将单卡修订与各数据集成本审计关联。

## 范围

固定 GSM8K 训练得到的全部 36 个 Gate 检查点、同一 Round 和各验证门限，
在三个目标数据集上生成 Direct、R1、R2、独立 Qwen3-14B 候选。
不在目标数据上训练、调阈值或选择新的主模型，不复用旧迁移轨迹。
当前入口只完成生成及审计，离线策略评测仍需单独适配，不能将生成完成称为实验全部完成。

| 数据集 | 主结果题数 | 问题去重 | 排除训练/验证文本重合并去重 |
| --- | ---: | ---: | ---: |
| GSM1K | 1205 | 1205 | 1201 |
| SVAMP | 300 | 300 | 300 |
| MultiArith | 180 | 178 | 178 |

完整集为主口径，不静默删除样本；另外报告提前锁定的敏感性子集。
重复题的统计区间需按规范化问题聚类重采样，不把重复记录当作独立观测。
这些数据集在历史阶段曾被查看，不声称是从未接触的确认集。文本去重不排除语义重合。

## 原始双卡启动记录（已由上述修订替代）

```bash
cd /path/to/adaptive_reflection
bash scripts/run_locked_math_transfer.sh --base-gpu 0 --reflector-gpus 0,2 --preflight-only
bash scripts/run_locked_math_transfer.sh --base-gpu 0 --reflector-gpus 0,2
```

两条命令分别为预检和实际生成；已经有后台任务时不要再次运行第二条。
中断后使用原命令续跑，逐题验证已完成调用及行记录，缺少调用或哈希不一致会停止。
`run_protocol.json` 冻结代码、启动脚本和显卡布局，原目录不允许静默切换布局或代码。
输出目录是 `outputs/math_transfer_locked_v1/`；本轮不使用旧脚本的 `GPU` 环境变量。

先在 GPU 0 加载 Base，处理所有目标问题后释放，再以 TP=2 在 GPU 0、2 加载强模型。
显存不足按既有规则等待，不杀其他进程；端口 8014 被占用则停止，不借用未知模型服务。
物理卡位与 GSM8K 测试不同，须披露硬件变化，不声称逐位等价。

## 生成与失败边界

保持训练协议的原始提示、模型、thinking 开关、temperature=0 和 seed=42。
上下文 32768，输出逐次上限 4096、8192、16384，安全余量 512。
恢复 length 是用原提示提高输出上限重新生成，不是续写；每次实际输入输出都计费。
R2 输入第一轮可见回答，不输入原始 thinking 文本。

- 输出预算耗尽：空预测、零系统作答得分，保留所有实际成本，不视为完整语义错误。
- Base 失败：R1/R2 标为上游阻断，只保留已支付 Base 成本；独立强模型仍执行。
- R1 失败：R2 不虚构一次调用，标为阻断并保留 Base+R1 费用；独立强模型仍执行。
- 无合法初始回答时，未来离线策略按锁定规则返回失败 Direct，不对伪造文本运行 Gate。
- 网络异常、未知结束原因、未闭合 thinking、窗口不足等其他错误停止任务，日志保留待处理。
  没有完成所有问题与所有可执行路径的核验，就不会导出完成审计。

不能保证模型永不截断；保证已识别的截断不被伪装为正常完成，并保留失败分母和费用。

## 产物

- `manifest.json`：数据与敏感性子集准备锁，不随运行状态更新。
- `run_protocol.json`、`generation_preflight.json`：运行协议和窗口预检查。
- `status.json`：实际运行阶段；`launcher.json`：后台启动信息（后台启动时创建）。
- `<dataset>/calls/`：每次请求/响应及付费 usage。
- `<dataset>/rows/base/`、`rows/qwen14/`：逐题不可覆盖结果。
- `<dataset>/policy_test_accounted.jsonl`、`accounting_audit.json`：全覆盖导出及费用审计。

`generation_complete_pending_offline_evaluation` 仅表示候选生成结束，不代表策略效果评测结束。
准备阶段 `PREPARATION.md` 是历史准备快照，不是实时状态。

## 本地测试

```bash
python -m unittest tests.test_prepare_locked_math_transfer tests.test_run_locked_math_transfer
```

测试覆盖重复/重合、完整路径累计成本、失败后依赖阻断、独立对照不被跳过和未知异常停止。
这些测试不替代真实模型执行及最终逐题审计。

## 匹配调用数随机 Gate 对照（2026-09-22）

在不产生新 LLM 调用的前提下，复用 `outputs/math_transfer_locked_v1` 的完整候选与冻结策略决策，新增匹配调用数随机对照：

```bash
python -u -m src.evaluate_transfer_random_baseline \
  --output outputs/math_transfer_locked_v1/evaluation/random_baseline_v1
```

`uniform_gate` 在全部题目中随机选择与策略相同数量的反思题；`round_stratified_gate` 进一步匹配策略选择的 R1/R2 数量。随机对照不读取标准答案或 token 成本来选题。结果位于 `outputs/math_transfer_locked_v1/evaluation/random_baseline_v1/REPORT.md` 和 `summary.json`。

该对照用于判断策略是否学到了“哪些题值得反思”，而不是只因为反思率更高而获得收益。相同调用数不等于相同 token，因此同时报告输入、输出和总 token；随机列为解析期望，逐题差异的 bootstrap 区间保存在 JSON 中。该分析仍基于已观察并用于开发的固定评测轨迹，属于探索性诊断，不构成独立确认显著性结论。
