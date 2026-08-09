# 旧 Qwen3 熵诊断图（仅供参考）

这些 PNG 来自历史实验：

- 学生模型：Qwen3-1.7B
- 方法：Vanilla OPD 诊断实验
- 训练长度：500 steps
- 原实验目录：
  `opd-alfworld-sync-repro/runs/2026-08-06_sync-vanilla-opd-tcod-mc-restart-after-full274/analysis`

它们不是当前 Qwen2.5-3B TCOD-F2B/Vanilla baseline 的结果，不能用于正式结果比较。
保留它们的目的，是为下一阶段研究提供熵变化、frontier、KL、成功率和 rollout
可变性等分析视角与画图样式参考。

## 图表内容

- `teacher_entropy_frontier_heatmap.png`：逐 rollout 的教师熵变化 frontier 热力图。
- `entropy_curve.png`：师生熵随 trajectory turn 的变化。
- `entropy_by_model_version.png`：熵随训练模型版本的变化。
- `teacher_entropy_by_outcome_progress.png`：按最终成功/失败分组的归一化进度曲线。
- `teacher_entropy_threshold_crossing_*.png`：不同熵阈值的首次持续越界时机。
- `teacher_entropy_observation_boundary.png`：新 observation 边界附近的教师熵。
- `teacher_entropy_rollout_variability.png`：rollout 间教师熵变化的可变性。
- `kl_surprisal_curve.png`、`reverse_kl_loss_curve.png`：KL、surprisal 与训练 loss。
- `mass_success_curve.png`：top-k mass、有效 action 与训练 rollout 成功率。

## 脚本与数据边界

绘图脚本、单元测试和当时的 workflow 诊断埋点保存在
[`research_tools/legacy_entropy_diagnostics/`](../../research_tools/legacy_entropy_diagnostics/)。

当前 baseline 训练日志没有逐 token teacher/student entropy，因此这些图没有在当前
Qwen2.5-3B 结果上重画。旧实验的 2.06GB 原始 `trajectory_metrics.jsonl` 仍只存在于
旧源目录，没有复制进 baseline；在决定删除旧源目录前，如需未来逐行重算这些旧图，
应再单独确认是否归档该原始 JSONL。
