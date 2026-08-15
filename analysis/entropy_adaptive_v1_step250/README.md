# Entropy-adaptive v1: corrected step-250 results

## 结论更正

v1 不是“逐题课程失败”，因为它没有实现逐题多轮课程。本次训练清洗后共有 2,736 条轨迹和 2,736 个不同的 `game_id`，没有任何题目被第二次采样；workflow 也没有保存 `game_id -> curriculum frontier` 的状态。它实际测试的是：对每条只出现一次的完整 rollout，临时计算一次 entropy frontier，只把此前缀送入 loss。

因此，这次实验应被解释为一次无状态、单遍的 adaptive loss-selection pilot。它达到三 seed 平均 72.63% 的 full274 成功率，说明这种很弱的第一版仍能学到相当多内容；但它不能验证或否定“同一道题反复学习并将 frontier 单调扩展到 30”的核心设想。

## Prompt 截断与绘图修正

旧图错误地把 `prompt_truncated` Experience 当成真实学生响应。累计 prompt 超过 10,240 tokens 后，当前 runtime 返回一个单 token、零 action-mask 的占位 Experience，而不是继续生成。workflow 仍把这个占位 token 送给教师打分，因此产生了人为的极低 teacher entropy。

去重后的 50,470 个 turn 中：

- 34,858 个是真实生成响应；
- 15,612 个是 prompt 占位行；
- 1,311/1,311 条失败训练轨迹最终都发生 prompt 截断；
- 最新 model version 220–250 的 104 条失败轨迹全部发生截断。

修正版默认图只用真实生成响应计算 entropy、surprisal、sampled reverse KL 和 action validity。热力图把占位区留白，并用灰色叉号标出首次 prompt 截断；原始受污染图片保留为 `*_contaminated.png`。

## v1 实际规则

每条轨迹前三个 turn 的 response-level teacher top-16 partial entropy 定义局部 baseline：

$$
B_i=\frac{1}{3}\sum_{t=0}^{2}H_{i,t}.
$$

检测第一个满足下式的三 turn 窗口末端：

$$
\frac{1}{3}\sum_{j=t-2}^{t}(H_{i,j}-B_i)\ge 0.175.
$$

环境和 teacher 仍处理完整轨迹，但 crossing turn 及其后缀不进入训练队列。这个 frontier 每次 rollout 都从零重算，不读取上一次同题进度，也不保证单调增长。

## 修正后的机制统计

421/2,736 条轨迹触发 frontier；其中 392 条在 frontier 位置或之后仍至少有一个真实生成响应，29 条在该位置已经只剩 prompt 占位行。在这 392 条可分析轨迹中：

- 354/392 = 90.31% 的真实后缀均值低于局部 crossing 峰值；
- 63/392 = 16.07% 的真实后缀均值低于前三 turn baseline；
- 154/392 = 39.29% 的最后三个真实响应均值低于 baseline；
- baseline、crossing、真实后缀和最后三响应的平均 entropy 分别为 0.2004、0.4421、0.3061 和 0.2538。

所以，“crossing 后长期保持高熵”和“crossing 后普遍恢复到低于初始 baseline”都不成立。更准确的描述是：局部峰值通常会回落，但后缀整体仍多半高于早期 baseline。相对 frontier 的真实响应样本数也从 turn 0 的 392 条降至 turn +15 的 36 条，极晚 turn 曲线仍有严重 survivorship bias。

## 全局训练过程

尽管没有逐题状态，v1 会因学生整体变强而在后续不同题目上自动放宽：平滑 frontier 触发率从早期约 53% 降到 step-250 附近 4.69%，保留 turn 比例从约 63% 升到 97.29%。这说明 entropy rule 形成了一个全局、隐式的 curriculum，但不是同题多轮 curriculum。

不能再声称“模型因为永远删掉失败题后缀，所以完全学不回后期状态”。更审慎的说法是：被触发题目的当次后缀被删除；后续不同题目仍可能提供相似状态，而且实际保留率很快升高。最终性能差距是否由硬截断造成，仍需要 matched full-loss control 才能作因果归因。

## Frozen full274 结果

三次评估都使用 horizon 30、累计 memory、严格 lowercase action parser、temperature 0.4、top-p 1、top-k -1 和 512 response tokens：

- seed 42：193/274 = 70.44%；
- seed 43：201/274 = 73.36%；
- seed 44：203/274 = 74.09%；
- 三 seed 平均 72.63%，样本标准差 1.93 percentage points。

共享 seed 42 下，Adaptive v1 比 Vanilla OPD 的 79.56% 低 9.12 pp，比 TCOD F2B 的 84.67% 低 14.23 pp。这个差距是真实的，但它比较的是“无状态单遍筛选”与两个基线，不是用户提出的完整逐题课程。

## v2 需要新增的核心机制

v2 若要真正检验逐题学习节奏，至少需要：

1. 让同一 `game_id` 在训练中重复出现，而不是单遍无放回采样；
2. 持久化每题的 `retained_frontier`、访问次数和最近 entropy profile；
3. 保证课程长度单调不减，例如

$$
k_i^{(r+1)}=\min\left(30,\max\left(k_i^{(r)},\hat f_i^{(r)}\right)\right),
$$

其中 $r$ 是题目 $i$ 的第 $r$ 次访问，$\hat f_i^{(r)}$ 是当次基于真实响应估计的安全 frontier；

4. 预先规定最终到达 30 的探索或保底扩展机制，否则某些题可能永远停在早期 frontier；
5. 在 frontier 统计前消除 prompt 占位行，并控制累计上下文长度；
6. 设置 matched full-loss 与原 TCOD 对照，使用相同 task visits、environment steps、teacher tokens 和 optimizer updates。

## 资产

- 修正版入口：`analyze_training_corrected.py`
- 修正前分析入口：`analyze_training.py`
- 修正摘要：`summary_nontruncated.json`
- 真实响应逐 step：`diagnostics_by_explorer_step_nontruncated.csv`
- 真实响应逐 turn：`diagnostics_by_trajectory_turn_nontruncated.csv`
- 真实响应 frontier 对齐：`frontier_aligned_rows_nontruncated.csv`
- 污染审计：`prompt_truncation_contamination_audit.png`
- 修正版训练总览：`training_overview.png`
- 修正版 outcome 曲线：`teacher_entropy_by_turn_outcome.png`
- 修正版 frontier 图：`frontier_mechanism.png`
- 修正版热力图：`teacher_entropy_frontier_heatmap_latest.png`
- 三 seed 评估：`evaluation_three_seed_comparison.png`
- seed-42 基线比较：`evaluation_seed42_frozen_baselines.png`

