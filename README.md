# TCOD ALFWorld 基线

本目录冻结了我们在 2026-08-09 确立的 Qwen2.5-3B / GiGPO-Qwen2.5-7B
ALFWorld 训练与评测机制。之后的开发和实验均应以本目录为起点。

需要注意：这是我们当前可复现、可对照的本地基线，并不表示所有实现细节都与论文完全一致。
我们要复现的论文是 [arXiv:2604.24005](https://arxiv.org/abs/2604.24005)。

## 论文主结果表

下表对应论文 Table 2：ALFWorld 上 TCOD 与 OPD 的分布内、分布外和 Hard 集结果。
SR 是成功率（%），Rounds 是每道题的平均 action 轮数。数值按论文原表抄录；为方便与
本地结果逐项核对，这里不重复标注论文中的相对 Vanilla 增减下标。

| 模型组 | 方法 | Valid Seen SR↑ | Valid Seen Rounds↓ | Valid Unseen SR↑ | Valid Unseen Rounds↓ | Hard SR↑ | Hard Rounds↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Teacher | Qwen2.5-7B-RL | 85.71 | 10.61 | 76.87 | 13.06 | 6.61 | 27.31 |
| Qwen2.5-3B Student | Zero-shot | 7.86 | 28.73 | 2.24 | 29.63 | 0.83 | 29.88 |
| Qwen2.5-3B Student | SFT | 32.14 | 22.85 | 25.37 | 24.16 | 4.96 | 29.12 |
| Qwen2.5-3B Student | Vanilla OPD | 65.72 | 14.73 | 60.45 | 16.21 | 10.74 | 28.64 |
| Qwen2.5-3B Student | TCOD-B2F（η=2） | 77.86 | 12.57 | 70.90 | 14.56 | 13.22 | 28.16 |
| Qwen2.5-3B Student | TCOD-F2B（η=2） | 81.43 | 11.76 | 79.19 | 12.47 | 9.92 | 28.57 |
| Qwen2.5-7B Student | Zero-shot | 9.29 | 28.34 | 8.96 | 28.46 | 1.65 | 29.77 |
| Qwen2.5-7B Student | SFT | 54.29 | 18.92 | 48.73 | 20.11 | 8.26 | 28.73 |
| Qwen2.5-7B Student | Vanilla OPD | 75.37 | 13.18 | 72.14 | 13.37 | 13.22 | 27.89 |
| Qwen2.5-7B Student | TCOD-B2F（η=2） | 86.43 | 11.06 | 77.61 | 13.16 | 20.66 | 27.07 |
| Qwen2.5-7B Student | TCOD-F2B（η=2） | 82.14 | 13.22 | 76.12 | 13.22 | 18.18 | 27.37 |

## 我们的复现结果

下表与论文主表使用完全相同的行列。`—` 表示目前没有运行，或者没有足以直接对应论文
该单元格的结果。

| 模型组 | 方法 | Valid Seen SR↑ | Valid Seen Rounds↓ | Valid Unseen SR↑ | Valid Unseen Rounds↓ | Hard SR↑ | Hard Rounds↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Teacher | GiGPO-Qwen2.5-7B-RL† | 96.43 | 8.38 | 91.04 | 10.51 | — | — |
| Qwen2.5-3B Student | Zero-shot† | 12.86 | 27.63 | 12.69 | 28.09 | — | — |
| Qwen2.5-3B Student | SFT | — | — | — | — | — | — |
| Qwen2.5-3B Student | Vanilla OPD step 250 | 82.14 | 11.26 | 76.87 | 12.73 | — | — |
| Qwen2.5-3B Student | TCOD-B2F（η=2） | — | — | — | — | — | — |
| Qwen2.5-3B Student | TCOD-F2B（η=2）step 250 | 87.14 | 10.35 | 82.09 | 11.87 | — | — |
| Qwen2.5-7B Student | Zero-shot | — | — | — | — | — | — |
| Qwen2.5-7B Student | SFT | — | — | — | — | — | — |
| Qwen2.5-7B Student | Vanilla OPD | — | — | — | — | — | — |
| Qwen2.5-7B Student | TCOD-B2F（η=2） | — | — | — | — | — | — |
| Qwen2.5-7B Student | TCOD-F2B（η=2） | — | — | — | — | — | — |

本地 Vanilla OPD 和 TCOD-F2B 两行是当前正式结果，使用完全相同的冻结评测协议：
完整 274 道题、环境 horizon 30、累计对话 memory、严格 action 解析、
temperature 0.4、seed 42，以及 512-token response 上限。对应的 task-weighted
overall 成功率分别是 218/274（79.56%）和 232/274（84.67%），TCOD 提升
5.11 个百分点。

† 教师与 Zero-shot 行来自修复第二步 prompt 后的历史 h50 逐题轨迹，并按前 30 步
重新统计 SR 和 Rounds；它们使用 512-token 上限，但没有启用当前正式评测的累计 memory
和严格 parser，因此用于记录已有结果，不应与 Vanilla/TCOD 两行作完全同协议的定量比较。
此外，本地教师是下载的 GiGPO RL 模型，并非论文的 GRPO teacher checkpoint。

## 冻结协议概览

- 学生模型：Qwen2.5-3B-Instruct
- 教师模型：GiGPO-Qwen2.5-7B-Instruct-ALFWorld
- TCOD 训练 workflow：`TCOD_f2b_alfworld_workflow`
- Vanilla 训练 workflow：`OPD_promptfix_alfworld_workflow`
- 训练步数：250
- 训练 batch size：64
- loss aggregation：`seq-mean-token-mean`
- 环境最大步数：30
- response 上限：512 tokens
- 不包含 SFT 阶段
- 评测 action 必须严格符合小写 `<action>...</action>` 格式

我们修复了公开代码中第二步 prompt 组装时丢失任务上下文的问题。该修复是本地基线的
一部分。教师模型选择、512-token 上限及其他与论文并非完全一致之处，详见
[`docs/BASELINE_SPEC.md`](docs/BASELINE_SPEC.md)。

## 目录结构

- `trinity/`：实际训练运行时代码，以及隔离后的冻结评测实现。
- `configs/train/`：可执行的 TCOD 和 Vanilla 训练配置；`.source.yaml` 文件保留
  历史运行配置，仅用于溯源。
- `configs/eval/`：可执行的 full274 冻结评测配置及其历史源配置。
- `configs/validation/`：用于验证最终 checkpoint 推理链路的最小配置。
- `models/`：Qwen2.5-3B 学生初始化模型和 GiGPO 7B 教师模型。
- `checkpoints/`：TCOD 与 Vanilla 完整 step-250 状态，包括优化器分片和
  Hugging Face 导出。
- `data/`：训练数据、Seen/Unseen 冻结清单和 ALFWorld runtime。
- `results/`：最终训练日志及支撑当前结论的完整评测结果。
- `analysis/frozen_full274_reproduction/`：由当前正式 full274 逐题结果重画的热力图、
  论文对比图、逐题矩阵、训练曲线、CSV 和输入哈希溯源。
- `analysis/reference_legacy_qwen3_entropy_diagnostics/`：旧 Qwen3-1.7B、500-step
  Vanilla OPD 熵诊断 PNG，仅作为后续研究参考，不属于当前 baseline 结果。
- `research_tools/legacy_entropy_diagnostics/`：隔离归档的旧版逐 token 熵诊断绘图、
  测试、埋点实现和参考图；参考图属于 Qwen3-1.7B 实验，不属于当前 baseline 结果。
- `validation/`：静态验证、端到端 smoke 结果、字节比较记录和 SHA256 清单。
- `paper/`：论文 PDF 和提取文本。
- `docs/`：基线规范、验证报告、来源与资产边界；`docs/upstream/` 保存上游说明，
  `docs/archive/` 只保存已明确标注、不会作为当前结论引用的历史研究文档。
- `.venv_tcod/`：已重定位到本目录的 Python 运行环境。

## 验证状态

当前 baseline 已完成以下验证：

- 3,553 条训练数据、140 条 Seen 和 134 条 Unseen 清单均可访问对应 PDDL 文件。
- TCOD 和 Vanilla step-250 checkpoint 均包含完整模型、优化器和恢复状态。
- 模型、checkpoint 与 buffer 已和源目录逐字节比较，结果一致。
- 122 个模型及 checkpoint 文件已生成 SHA256 校验清单。
- TCOD 和 Vanilla 均从本目录成功加载 final checkpoint，并完成 Seen + Unseen
  端到端推理、严格解析、ALFWorld 交互和结果汇总。
- 正式 full274 历史结果已在本目录重新核对：TCOD 为 232/274，Vanilla 为
  218/274。

完整验证证据见 [`docs/VALIDATION.md`](docs/VALIDATION.md)。

## 常用命令

运行不占用 GPU 的完整性检查：

```bash
bash scripts/validate_baseline.sh
```

在 tmux 中启动新的四卡训练：

```bash
bash scripts/launch_train_tmux.sh tcod 4,5,6,7
bash scripts/launch_train_tmux.sh vanilla 0,1,2,3
```

在 tmux 中启动冻结的 full274 评测：

```bash
bash scripts/launch_eval_tmux.sh tcod 0,1,2,3
bash scripts/launch_eval_tmux.sh vanilla 0,1,2,3
```

重新校验全部大模型和 checkpoint 文件（约需读取 111GB）：

```bash
sha256sum -c validation/MODEL_CHECKPOINT_SHA256SUMS
```

用冻结的 TCOD/Vanilla full274 结果重新生成研究图表：

```bash
bash scripts/run_baseline_plots.sh
```

全部当前图表与历史参考图的分区索引见
[`analysis/README.md`](analysis/README.md)。
生成物和口径说明见
[`analysis/frozen_full274_reproduction/README.md`](analysis/frozen_full274_reproduction/README.md)。
正式 baseline、Adaptive v1、full-loss 对照、two-stage、task-matched Vanilla、
未完成运行和历史迁移资产的统一状态见
[`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md)。

## 后续开发规则

- 所有训练任务必须通过 tmux 启动。
- 除非某次实验在配置或实验 README 中事先明确声明豁免，后续训练默认必须启用并保留
  画图所需的逐 turn、逐 token 师生诊断数据。至少应设置
  `diagnostics_enabled: true`、正数 `diagnostics_top_k`（通常为 16）、
  `diagnostics_required: true`，并将 `diagnostics_path` 指向该实验自身目录下完整的
  `trajectory_metrics.jsonl`。
- 必须同时保留 launcher log、TensorBoard events、实际运行配置、模型版本与任务结果
  标识，以及 entropy、surprisal/KL、token-block 等字段；训练结束后不得默认删除或
  截断。若确需豁免，必须在启动前记录原因以及因此无法生成的图表。
- 不要直接修改冻结的训练配置、评测配置、checkpoint 或正式结果。
- 新实验应复制配置到 `configs/experiments/`，并使用新的输出目录。
- 所有新代码、配置、日志、checkpoint 和结果都应保存在本目录内。
- 删除 `opd-alfworld-sync-repro` 或 `tcod-f2b-repro` 属于单独操作，必须在确认
  本 baseline 无遗漏后再讨论和执行。

本机专用的 `AGENTS.md` 保留在工作目录但不会提交。基线规范见
[`docs/BASELINE_SPEC.md`](docs/BASELINE_SPEC.md)，来源与 Git/本地资产边界见
[`docs/PROVENANCE.md`](docs/PROVENANCE.md)，原始上游英文 README 保存在
[`docs/upstream/TRINITY_README.md`](docs/upstream/TRINITY_README.md)。
