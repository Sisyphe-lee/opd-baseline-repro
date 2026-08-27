# 多 seed 战役最终汇总（2026-08-16 至 08-19，21 个 250 步训练）

## 终点总榜（step250 冻结评测协议；★=终点为 step246，explorer 配额先竭）

| 方法 | seeds (A100 合作者 + B200 本项目) | n | mean±std | min |
|---|---|---|---|---|
| adaptive τ=0.100 | 238,227 + 230,234,206,221,228,222 | 8 | **225.8±9.8** | 206 |
| TCOD-F2B | 232 + 223,224,226,219,223,213 | 7 | 222.9±5.9 | 213 |
| Vanilla OPD | 218 + 224,216,211,173,195★ | 6 | 206.2±19.0 | 173 |

统计（Welch / Mann-Whitney 双侧）：
- adaptive vs Vanilla: Δ+19.6, p=0.054 / **0.020**
- TCOD vs Vanilla: Δ+16.7, p=0.085 / **0.045**
- adaptive vs TCOD: Δ+2.9, p=0.50 / 0.39（等价）

## 主结论
1. **两个课程方法显著优于 Vanilla**（非参检验），且 Vanilla 的病理即论文主张的
   不稳定性：6 seed 中 3 个坍塌（211/195/173）。
2. **adaptive 与 TCOD 终点等价**；adaptive 的价值主张 = 免调参（无 η/k_start）+
   仅训 ~70-80% turn（省 trainer 算力）达到同等均值；短板 = 末段方差大（σ9.8 vs 5.9）。
3. **延长无益于 adaptive、微益于 TCOD**（EXTENSION_310_RESULTS.md）。
4. **checkpoint 汤（220/240/250 平均）= 有效的稳定化**：5 汤 vs 5 端点：
   μ 223.8→222.0（−1.8），σ 11.0→6.3，worst 206→216；两份 203 汤均 ≥ 各自三端点均值
   （218 vs 217.7；232 vs 227）。v2 设计原则一号候选。
5. **训练加速在 V0 栈上全线证伪**：fast8 布局（同 seed −26）、prefix cache（+6% 不值）、
   CUDA graph（1.38× 提速但同 seed −14，越出 TCOD 臂 3.6σ）——凡触碰 rollout
   数值/并发/时序皆改变训练结果。评测 fast8 (11×) 经逐题验证无害、已采纳。
   训练提速唯余 vLLM 升级路线（需全面重验）。

## 运维备忘
- exit_status 不可信（actor 崩溃/队列关闭均可能写 0），以 checkpoint 完整性为准。
- explorer/trainer 何者先达 250 有随机性；终点以 trainer 最终 checkpoint 为准并记步号。
- 已修 launcher 端口自冲突；vanilla OOM→packing 12288（审计证明梯度中性）。
