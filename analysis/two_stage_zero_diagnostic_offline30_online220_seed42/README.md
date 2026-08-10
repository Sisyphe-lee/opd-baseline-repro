# Offline-30 + online-220 零号诊断

## 结论

这次实验支持“离线阶段能够把初始学生加热”，但不支持“简单的两阶段硬切换足以替代 TCOD”。

- Student init：27/274（9.85%）。
- Offline 30：102/274（37.23%），相对 init 增加 75 题、27.37pp；逐题 McNemar
  双侧精确检验 $p=2.17\times10^{-16}$。
- Offline 30 + vanilla online 220：201/274（73.36%），相对 offline30 增加
  99 题、36.13pp；$p=5.05\times10^{-20}$。
- 最终模型仍低于冻结 Vanilla OPD step250 的 218/274（少 17 题、-6.20pp，
  $p=0.0300$），也低于 TCOD step250 的 232/274（少 31 题、-11.31pp，
  $p=5.54\times10^{-6}$）。

因此，offline warm-start 明显缓解了弱学生的 early collapse，但目前证据表明 TCOD 的收益
不只是冷启动补偿。这个判断仍受预算不完全匹配限制：三者都是 250 个 optimizer update，
但本实验把其中 30 步用于离线 CE、220 步用于在线 OPD；teacher trajectory 采集成本、
teacher-scored token、student rollout token、GPU layout 和 wall-clock 均未与两个冻结 baseline
严格匹配。

## 实验定义

- 冻结评测：full274、horizon 30、temperature 0.4、seed42、response 512、修复 step-2
  prompt、累计 memory、严格小写 `<action>...</action>`。
- 离线数据：teacher 在全部 3,553 个训练游戏上生成轨迹；成功 2,943 条（82.83%）；所有
  成功轨迹的前缀共形成 21,692 条训练样本。
- Offline 30 是 teacher-success response 上的 hard-label SeqKD/CE（等价于这里所说的
  SFT/behavior cloning 阶段），**不是**缓存 teacher 全词表 logits 的 soft offline KL。
- Online 220 从 offline30 权重开始，使用 vanilla full-loss OPD；没有 entropy cutoff。
- Online GPU layout：2 student rollout + 4 teacher + 2 trainer。训练从 17:22:45 到
  22:25:57 UTC，约 5 小时 3 分。

## Frozen full274

| 阶段 | Seen | Unseen | Overall | 平均环境轮数 | Parse valid | Admissible | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| Student init | 14/140 | 13/134 | 27/274 (9.85%) | 27.77 | 61.11% | 49.05% | 90.15% |
| Offline 30 | 53/140 | 49/134 | 102/274 (37.23%) | 21.57 | 75.24% | 64.94% | 62.77% |
| Offline 30 + online 220 | 110/140 | 91/134 | 201/274 (73.36%) | 13.34 | 89.94% | 85.15% | 26.64% |

Warm 不只体现在最终成功率：offline30 同时降低 timeout、重复动作和 unchanged observation，
并提高严格 action parsing 与 admissibility。因此它确实改善了 student-on-policy occupancy，
而不只是降低 teacher-forced loss。

## 在线诊断

最终干净运行记录了 151 个 Explorer collection step、2,416 条轨迹和 40,295 个 turn row。

- 11,340/40,295（28.14%）是累计 prompt 超过 10,240 后产生的单 token、zero-mask
  `prompt_truncated` placeholder；真实响应为 28,955 行。所有 entropy/KL 图均显式排除
  placeholder，原始 JSONL 保持不变。
- 1,439 条成功轨迹均未截断，平均 7.63 个真实 turn；977 条失败轨迹全部最终截断，截断前
  平均 18.39 个真实 turn。因而失败长尾仍受 runtime prompt cap 污染，不能把 placeholder
  当作有效 supervision 或真实低熵响应。
- post-hoc early bin（student model version 0–30）到 late bin（189–218）：rollout success
  从 36.01% 升到 73.91%，timeout 从 63.99% 降到 26.09%，平均环境轮数从 21.88 降到
  13.31，token-weighted sampled reverse KL 从 0.2507 降到 0.1281。
- 同期 placeholder row 比例从 34.15% 降到 22.54%。这说明 occupancy 在改善，但 prompt cap
  仍是显著的无效 teacher/环境计算来源。

这里的 entropy 是 response-level top-16 partial/head entropy，不是全词表归一化 entropy；
KL 是 student-sampled token 上的 sampled reverse KL 估计，不是完整词表 KL。

## Checkpoint 与运行完整性

- Offline：`global_step_30` 保存了完整 FSDP/optimizer state 和 HF export。
- Online：`global_step_{20,40,...,220}` 共 11 个断点；每个均包含 `.full_checkpoint`、
  两个 model rank、两个 optimizer rank 和两片 HF safetensors，单个约 42GB。
- Trainer 日志包含且仅包含完成的 step 1–220，无缺号；最终 step220 正常保存并退出。
- 第一次 online 启动因缺少 SQLite buffer 父目录在早期失败；证据完整保存在
  `runs/experiments/two_stage_distillation/failed_attempts/online_attempt1_missing_buffer_parent/`。
  修复后从 offline30 重新启动了干净的 220-step run，未混入失败尝试的数据或权重。

## 产物

- `summary.json`：三阶段 full274 与逐题配对检验。
- `stage_metrics.csv`、`paired_comparisons.csv`、`reference_comparisons.csv`：可审计表格。
- `success_rates.png`、`behavior_metrics.png`：warm 与最终能力图。
- `online_diagnostics/`：entropy、KL、success、timing 图表及逐 step/trajectory CSV。
- 原始 full274：
  `runs/experiments/two_stage_distillation/evaluation/{student_init_seed42,offline30_seed42,offline30_online220_seed42}/`。
- 原始在线诊断：
  `runs/experiments/two_stage_distillation/diagnostics/offline30_online220_seed42/trajectory_metrics.jsonl`。

## 下一步解释

这轮最有价值的作用是把两个问题拆开了：

1. “离线成功轨迹能否解决初始学生完全不会交互？”——能，证据很强。
2. “解决冷启动后，vanilla OPD 是否足以达到 TCOD？”——本轮不能；最终仍显著落后。

下一轮若研究 soft-logit offline KD，应将它视为新的 ablation，而不是声称本轮已经验证；同时必须
固定 teacher-scored token、student rollout token 或总 GPU-hour，才能判断 soft logits 是否比
hard-label response 学习更高效。
