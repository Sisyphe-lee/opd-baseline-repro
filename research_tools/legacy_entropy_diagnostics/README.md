# 旧版逐 token 熵诊断工具（隔离归档）

这里保存了旧目录中的熵曲线、teacher-entropy frontier 热力图、KL 曲线及其
生成脚本、单元测试和 workflow 埋点实现，避免后续清理源目录时丢失研究工具。

必须明确：`reference_figures/` 来自 **Qwen3-1.7B、500-step Vanilla OPD**
诊断实验，不是当前 Qwen2.5-3B TCOD-F2B baseline 的结果，不能混入正式复现表。

当前 Qwen2.5-3B TCOD/Vanilla 最终训练没有保存该脚本所需的逐 turn、逐 token
teacher/student top-k entropy JSONL，因此不能只用现有 checkpoint 和 launcher log
原样重画这些 entropy 图。后续实验需要把
`instrumentation/OPD_workflow_instrumented.py` 中的诊断埋点以新实验配置接入，重新
产生 `trajectory_metrics.jsonl`，再执行：

```bash
bash research_tools/legacy_entropy_diagnostics/scripts/plot_vanilla_opd_diagnostics.sh \
  --diagnostics /path/to/trajectory_metrics.jsonl \
  --output-dir /path/to/new/analysis
```

## 后续训练的默认保留规则

除非实验配置或实验 README 在启动前明确声明豁免，后续训练必须默认启用该诊断埋点并
完整保留 `trajectory_metrics.jsonl`。推荐的 workflow 参数为：

```yaml
diagnostics_enabled: true
diagnostics_top_k: 16
diagnostics_required: true
diagnostics_path: /path/inside/this/experiment/diagnostics/trajectory_metrics.jsonl
diagnostics_token_block_size: 4
diagnostics_store_token_ids: true
diagnostics_store_text: true
```

同时保留实际运行配置、launcher log、TensorBoard events、checkpoint/模型版本映射和
画图输出的 provenance。不得在训练结束后默认删除或截断这些文件。豁免说明必须写明
原因，以及 entropy frontier、turn curve、KL/surprisal 等哪些图将因此无法生成。

旧实验的 2.06GB 原始 `trajectory_metrics.jsonl` 没有复制到 baseline；本目录保留
脚本、测试、汇总 CSV、`summary.json`、分析说明和已生成的 PNG。若决定删除旧源目录
且仍需逐行复算旧 Qwen3 图，应先单独确认是否一并归档该原始 JSONL。
