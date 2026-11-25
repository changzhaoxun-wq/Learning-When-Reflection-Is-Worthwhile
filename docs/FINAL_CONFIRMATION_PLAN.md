# 最终确认实验与论文定稿清单

## 已锁定主策略

- 主任务：GSM8K。
- Gate 配置：`seed42_weighted_ce_control_original_round_0.10`。
- 选择来源：GSM8K 验证协议，不使用迁移测试集标签。
- 反思器：当前锁定迁移协议中的 Qwen3-14B。
- 新的 LLM 调用：0；当前迁移结果全部由已保存轨迹离线重放。

主策略在迁移数据集上的结果见：
`outputs/math_transfer_locked_v1/evaluation/prelocked_policy_v1/REPORT.md`。

## 已完成证据

- GSM8K 全量测试及失败成本核算。
- GSM8K 三随机种子 Gate 结果。
- 匹配调用数随机 Gate 对照。
- GSM1K、SVAMP、MultiArith 的固定策略迁移重放。
- 迁移修复/引错/Gate 漏修复分析。
- 输入、输出和总 token 的质量—成本统计。

## 独立确认要求

独立确认不能使用已经观察过的 GSM8K、GSM1K、SVAMP 或 MultiArith 测试结果。运行前冻结：

1. 主策略 checkpoint、Gate 阈值和 Round checkpoint。
2. Base/Reflector 模型、prompt、temperature、thinking 设置和输出上限。
3. 答案抽取、失败计分和 token accounting 规则。
4. 统计指标和主要比较对象。

确认集只运行一次，至少报告 Direct、Locked Policy、Fixed R1 和 matched-call random Gate。若没有新的独立题目，当前结果应标为 exploratory，而不能写成 confirmatory。

## 论文主张边界

- 主结论：自适应策略在 GSM8K 上能够用少量反思提高弱基础模型准确率，并相对固定反思降低平均 token。
- 迁移结论：GSM1K 上存在有限迁移；SVAMP 上泛化不足；MultiArith 主要体现低调用率下维持饱和准确率。
- 不声称：所有数据集普遍迁移成功、严格等 token 优势、端到端延迟等比例下降、或达到强模型等效准确率。

