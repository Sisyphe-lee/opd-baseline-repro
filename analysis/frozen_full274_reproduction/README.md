# 冻结基线复现图表

本目录由 `scripts/plot_baseline_reproduction.py` 直接读取两份冻结的 full274 逐题结果生成。

## 核心数值

- Vanilla：Seen 82.14%，Unseen 76.87%，Overall 79.56%。
- TCOD-F2B：Seen 87.14%，Unseen 82.09%，Overall 84.67%。
- 逐题配对：共同成功 208，仅 TCOD 成功 24，仅 Vanilla 成功 10，共同失败 32。

## 图表边界

- `task_type_success_heatmap.png`、`paired_outcome_heatmap.png` 和 `per_task_outcome_matrix.png` 均来自当前严格 512-token full274 评测。
- `training_rollout_comparison.png` 来自最终训练 launcher log；它是随机训练 batch 的在线指标，不是正式评测。TCOD 日志仅覆盖恢复后的 model version 80–248。
- 旧目录的 teacher-entropy frontier 热力图来自 Qwen3-1.7B、500-step 的另一项 Vanilla OPD 诊断实验，不能标为当前 Qwen2.5-3B TCOD baseline 的复现图。当前最终训练没有保存逐 token teacher/student entropy，因此无法仅靠现有 checkpoint 和日志原样重画该图。

旧图已作为非 baseline 参考独立保存在 [`../reference_legacy_qwen3_entropy_diagnostics/`](../reference_legacy_qwen3_entropy_diagnostics/)，相应绘图脚本、测试和埋点代码保存在 [`../../research_tools/legacy_entropy_diagnostics/`](../../research_tools/legacy_entropy_diagnostics/)。
