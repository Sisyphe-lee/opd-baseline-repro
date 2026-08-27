# B200 移植验证与多 seed 复跑（2026-08-16）

一天内在两台 8×B200 机器（umb-b200-158 / umb-b200-203）上完成的四个 250 步训练
及其 full274 冻结评测。训练配置：158 双基线为冻结 `configs/train/*.yaml` 原样
（诊断豁免记录见 `runs/training/README_B200_REPRO.md`）；203 双 adaptive 为
`configs/experiments/entropy_adaptive_v1_t0100_b203_repro_seed{42,43}_250step_4gpu_s1t1_r4.yaml`
（诊断齐全）。评测均为冻结协议（274 题、h30、temp 0.4、seed 42、512 tok、严格 parser）。
环境差异（B200/sm100、torch 2.8.0+cu128、flash-attn 2.8.1 自编、vLLM 0.10.2 V0 经
PTX JIT）见 `runs/training/README_B200_REPRO.md`。

## 结果

| 实验 | 机器 | Seen | Unseen | 总计 |
|---|---|---:|---:|---:|
| TCOD-F2B 复跑 | 158 | 117/140 (83.57%) | 106/134 (79.10%) | **223/274 (81.39%)** |
| Vanilla OPD 复跑 | 158 | 117/140 (83.57%) | 107/134 (79.85%) | **224/274 (81.75%)** |
| adaptive τ=0.100 seed42 复跑 | 203 | 125/140 (89.29%) | 105/134 (78.36%) | **230/274 (83.94%)** |
| adaptive τ=0.100 seed43（新种子） | 203 | 120/140 (85.71%) | 114/134 (85.07%) | **234/274 (85.40%)** |

## McNemar 配对检验（本目录 CSV）

| 对比 | 计数 | p | 判定 |
|---|---|---:|---|
| 冻结 TCOD (232) vs B200 TCOD (223) | 26/17 不一致 | 0.222 | 等价 |
| 冻结 Vanilla (218) vs B200 Vanilla (224) | 19/25 不一致 | 0.451 | 等价 |
| B200 TCOD (223) vs B200 Vanilla (224) | 24/25 不一致 | 1.000 | 打平 |
| （参照）冻结 TCOD vs 冻结 Vanilla | 10/24 不一致 | 0.024 | 冻结对显著 |

## 结论

1. **硬件移植等价性成立**：两条基线的 B200 复跑与冻结结果均无显著差异。
2. **adaptive τ=0.100 的真实水平**：四个独立训练（238/227/230/234）均值
   232.3/274 ≈ 84.8%，σ≈4.6 题；86.86% 与 82.85% 分别是同一分布的高/低抽样。
3. **方法论警示**：TCOD 相对 Vanilla 的 +14 题冻结优势在复跑对上消失
   （−1 题，p=1.0）。单 run 的方法间比较不足以支撑 <2pp 的差距主张；
   后续所有方法结论都应基于每臂 ≥3 个训练 seed。
4. 训练耗时（4 卡 B200）：TCOD 3.5h、Vanilla 6h、adaptive ≈4.5h；
   full274 评测 ≈40 分钟（4 引擎）。

## 溯源

- 158 评测产物：`runs/ckpt_store/evaluation/b158_repro_{tcod_f2b,vanilla_opd}_step250_full274/`
- 203 训练与评测产物：`/mnt/scratch/local-tianhej/opd-assets/runs/experiments/entropy_adaptive_v1_t0100_b203_repro_seed{42,43}_250step_4gpu_s1t1_r4/`（203 机器）
- 冻结参照：`results/evaluations/2026-08-0{8,9}_*-full274-h30-accmemory-strict/`
- 已知问题：`scripts/_run_entropy_experiment.sh` 端口推导在 RAY_PORT=16920 等取值下
  client 端口落入自身 worker 范围导致启动失败（16921 安全），待上游修复。
