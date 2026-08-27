# 多 seed 逐 checkpoint 曲线（2026-08-17）

图：`multiseed_curves.png`；数据：`curves_merged.csv`（B200 四条 lane 的 52 个
full274 点 + 合作者 A100 机器的三条冻结曲线）。生成脚本 `build_curves.py`。
所有点均为冻结评测协议（274 题 / h30 / temp 0.4 / eval seed 42 / 严格 parser）。
曲线方法学已验证：merger 导出与 trinity 最终导出逐张量一致（vanilla step250 435/435）。

## 两种估计量下的方法对比

**终点（step250）**：adaptive 238/230/234（A100/B200s42/B200s43），TCOD 232/223，
Vanilla 218/224 —— adaptive 领先 ~8-13 题。

**后期均值（step160-250 六点平均，方差更小）**：

| 方法 | run 级后期均值 | 合并 |
|---|---|---:|
| adaptive | 222.5 / 217.7 / 221.5 | **220.6 (80.5%)** |
| TCOD | 224.0 / 215.7 | 219.8 (80.2%) |
| Vanilla | 221.5 / 193.8 | 207.7 (75.8%)† |

† Vanilla 两 run 后期均值相差 27.7 题，run 间方差极大。

## 关键观察

1. **按后期轨迹平均，adaptive 与 TCOD 打平**（220.6 vs 219.8）；adaptive 的
   终点优势主要来自一个系统性的"末段跳变"。
2. **adaptive 的 step250 终点在全部 3 个 run 中都高出自身 step160-240 均值
   +12~+15 题**（TCOD 约 +7/+8，Vanilla 不稳定）。3/3 的一致性说明这不是抽签，
   而是方法在训练末段的某种真实机制（或与结尾队列排空/staleness 变化的交互），
   是下一个应当研究的现象——合作者未收口的 250→310 延长实验正好切题。
3. adaptive 的三条曲线彼此重叠度最高（后期均值极差 4.8 题 vs Vanilla 的 27.7），
   **稳定性本身是 adaptive 当前最硬的优势**。
4. fast8（8 卡 s2t4_r16 + 0.85 显存）训练的三个 run（tcod 217、vanilla 215、
   adaptive 210）全部低于同方法 4 卡 run，疑似布局对训练质量有 ~2-5pp 代价；
   解耦实验在跑（158: tcod/vanilla 4gpu seed43；203: adaptive fast8 seed42），
   结论出来前 fast8 布局不用于正式训练（评测侧 fast8 已验证无害，p=1.0）。
