# Analysis 图表索引

本目录把“当前 baseline 结论”和“历史研究参考”严格分开：

- [`frozen_full274_reproduction/`](frozen_full274_reproduction/)：当前
  Qwen2.5-3B Vanilla/TCOD-F2B 的正式冻结结果及据此生成的图表。输入协议是
  full274、horizon 30、累计 memory、严格 action parser、response 512。
- [`reference_legacy_qwen3_entropy_diagnostics/`](reference_legacy_qwen3_entropy_diagnostics/)：
  旧 Qwen3-1.7B、500-step Vanilla OPD 诊断实验的 PNG，仅用于研究思路和画图样式参考，
  不属于当前 baseline 结果，不能与正式复现数值混用。

当前正式结果可运行 `bash scripts/run_baseline_plots.sh` 重画。旧熵图不在当前结果上
重画，因为本次最终训练没有记录所需的逐 turn、逐 token 师生 entropy 数据。
旧版绘图脚本、测试和诊断埋点保存在
[`research_tools/legacy_entropy_diagnostics/`](../research_tools/legacy_entropy_diagnostics/)。

后续训练默认必须启用并保留上述绘图诊断数据，除非对应实验在启动前明确记录豁免原因
及会缺失的图表；详细字段和配置要求见绘图工具目录的 README 与根目录 `AGENTS.md`。
