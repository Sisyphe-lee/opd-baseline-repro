# 试验结果总索引

最后更新：2026-08-11。

本文档是本仓库试验结果的统一入口。它区分：

1. **冻结正式基线**：用于后续方法比较的固定参照；
2. **已完成研究试验**：有 checkpoint 和冻结评测，但不自动升级为正式基线；
3. **未完成运行记录**：没有完成 checkpoint 或冻结评测，不能作为试验结论引用；
4. **历史迁移资产**：仅用于溯源，不与当前 Qwen2.5-3B 结果混算。

除非特别注明，当前可比评测均采用 full274、环境 horizon 30、累计对话 memory、
严格小写 `<action>...</action>` 解析、temperature 0.4、response 上限 512 tokens。

## 一、冻结正式基线

| 方法 | Checkpoint | Seed | Seen | Unseen | Overall | 状态 |
|---|---|---:|---:|---:|---:|---|
| Vanilla OPD step 250 | [`checkpoints/vanilla_opd_step250/`](../checkpoints/vanilla_opd_step250/) | 42 | 115/140 | 103/134 | **218/274（79.56%）** | 正式基线 |
| TCOD-F2B step 250 | [`checkpoints/tcod_f2b_step250/`](../checkpoints/tcod_f2b_step250/) | 42 | 122/140 | 110/134 | **232/274（84.67%）** | 正式基线 |

原始训练记录位于 [`results/training/`](../results/training/)，冻结评测位于
[`results/evaluations/`](../results/evaluations/)。详细协议和验证证据分别见
[`BASELINE_SPEC.md`](BASELINE_SPEC.md) 与 [`VALIDATION.md`](VALIDATION.md)。

## 二、已完成研究试验

### 2.1 Entropy-adaptive v1，step 250

- 运行目录：[`runs/experiments/entropy_adaptive_v1_step10_8gpu_s2t4_r16/`](../runs/experiments/entropy_adaptive_v1_step10_8gpu_s2t4_r16/)
- 续训辅助记录：[`runs/experiments/entropy_adaptive_v1_resume_step10_to250_suffix/`](../runs/experiments/entropy_adaptive_v1_resume_step10_to250_suffix/)
- 完整分析：[`analysis/entropy_adaptive_v1_step250/README.md`](../analysis/entropy_adaptive_v1_step250/README.md)

| Seed | Seen | Unseen | Overall |
|---:|---:|---:|---:|
| 42 | 110/140 | 83/134 | 193/274（70.44%） |
| 43 | 109/140 | 92/134 | 201/274（73.36%） |
| 44 | 109/140 | 94/134 | 203/274（74.09%） |
| 三 seed 平均 | — | — | **72.63%** |

结论：试验完整，但结果弱于冻结基线。seed 42 下比 Vanilla OPD 低 9.12 个百分点，
比 TCOD-F2B 低 14.23 个百分点。v1 实际是逐 rollout、无状态的 retrospective
loss-prefix selection，不是同一题目多次访问并逐步扩展 frontier 的课程学习。

### 2.2 Adaptive/full-loss step-10 机制对照

- Adaptive 数据与评测包含在 v1 主运行目录中。
- Full-loss 对照：[`runs/experiments/entropy_full_v1_step10_8gpu_s2t4_r16_retry4/`](../runs/experiments/entropy_full_v1_step10_8gpu_s2t4_r16_retry4/)
- 配对分析：[`analysis/entropy_adaptive_v1_step10_pilot/README.md`](../analysis/entropy_adaptive_v1_step10_pilot/README.md)

| Seed | Adaptive | Full loss | 差值（题数） |
|---:|---:|---:|---:|
| 42 | 24/274 | 26/274 | -2 |
| 43 | 20/274 | 26/274 | -6 |
| 44 | 25/274 | 21/274 | +4 |
| 描述性合计 | 69/822（8.39%） | 73/822（8.88%） | -4 |

结论：10-step pilot 没有显示 adaptive 截断优于 full loss；三个 seed 的方向也不一致。
它只用于机制检查，不能替代 step-250 matched control。

### 2.3 Two-stage distillation

- 运行目录：[`runs/experiments/two_stage_distillation/`](../runs/experiments/two_stage_distillation/)

| 阶段 | Seed | Overall |
|---|---:|---:|
| Student initialization | 42 | 27/274（9.85%） |
| Teacher-success offline 30 steps | 42 | 102/274（37.23%） |
| Offline 30 + online 220 steps | 42 | **201/274（73.36%）** |

结论：完整训练产生了最终 checkpoint，但 seed 42 最终结果低于 Vanilla OPD 6.20 个
百分点、低于 TCOD-F2B 11.31 个百分点。目前只有一个评测 seed，不应把这一差距解释为
稳定的总体效应。

### 2.4 Task-matched Vanilla OPD，step 276

- 运行目录：[`runs/experiments/vanilla_opd_taskmatched_step276_8gpu/`](../runs/experiments/vanilla_opd_taskmatched_step276_8gpu/)
- 有效评测：[`evaluation/full274_seed42_8gpu/summary.json`](../runs/experiments/vanilla_opd_taskmatched_step276_8gpu/evaluation/full274_seed42_8gpu/summary.json)

seed 42 为 217/274（79.20%），其中 Seen 113/140、Unseen 104/134。相对冻结的
Vanilla OPD step 250 少 1 题（-0.36 个百分点），没有显示增加 task-matched 更新数带来收益。

## 三、未完成运行记录

下列目录没有 `global_step_*` checkpoint，也没有完整冻结评测 summary，**不得引用为
试验结果**。它们仅保留配置、启动日志、少量诊断或数据库，用于故障追溯：

| 目录 | 判定 |
|---|---|
| `entropy_adaptive_v1_step10_8gpu` | NCCL 初始化失败；无训练 step |
| `entropy_adaptive_v1_step10_fast` | 未完成；无 checkpoint/评测 |
| `entropy_adaptive_v1_step30` | 未完成；无 checkpoint/评测 |
| `entropy_full_v1_step10_8gpu_s2t4_r16` | 未完成；无 checkpoint/评测 |
| `entropy_full_v1_step10_8gpu_s2t4_r16_retry1` | 未完成；无 checkpoint/评测 |
| `entropy_full_v1_step10_8gpu_s2t4_r16_retry2` | 未完成；无 checkpoint/评测 |
| `entropy_full_v1_step10_8gpu_s2t4_r16_retry3` | 未完成；无 checkpoint/评测 |
| `entropy_full_v1_step10_fast` | 未完成；无 checkpoint/评测 |
| `entropy_full_v1_step30` | 未完成；无 checkpoint/评测 |

已经确认无效并删除的目录：

- `entropy_adaptive_v1_step10_8gpu_s1t5`：NCCL `Invalid rank requested: 1/1`；
- `entropy_adaptive_v1_step10_8gpu_s2t4`：只完成 step-0 权重同步，未完成 step 1。

## 四、历史迁移资产

[`runs/experiments/legacy_imports/2026-08-11_pre_cleanup/`](../runs/experiments/legacy_imports/2026-08-11_pre_cleanup/)
保存从 `opd-alfworld-sync-repro` 和 `tcod-f2b-repro` 迁移的代码快照、Git bundle、
历史评测、HF 导出、诊断和 manifest。这里包含 Qwen3-1.7B、不同 horizon、不同 prompt
协议等历史结果；除非单独完成协议对齐，否则不能与上面的当前 Qwen2.5-3B 表格直接比较。

## 五、结果引用规则

- “正式 baseline”只指第一节的 Vanilla OPD step 250 与 TCOD-F2B step 250。
- “完成试验”必须至少同时存在实际运行配置、训练日志、目标 checkpoint 和对应评测 summary。
- 多 seed 平均不能把同一批 274 个任务简单当作 822 个独立任务进行显著性检验。
- 失败/中止目录中的日志可以用于诊断，但不能报告为模型性能。
- 每个新试验必须写入独立的 `runs/experiments/<experiment_name>/`，并保留配置、日志、
  diagnostics、checkpoint、评测逐题记录和 summary。
- 更新本索引时以原始 `summary.json` 为准；图表或聊天中的手抄数字不是权威数据源。
