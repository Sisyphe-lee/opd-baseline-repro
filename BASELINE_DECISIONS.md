# Prefill-Aware OPD：当前研究与基线决策

更新时间：2026-07-30

本文是当前研究主线的 authority。2026-07-29 以前基于 OPSD、Qwen3-4B/8B、
LongICLBench BANKING77 的实验保留为历史探索，不再定义当前 baseline。

## 1. 研究问题

标准 On-Policy Distillation（OPD）在 student 自己生成的 response trajectory 上，
让 student 对齐 frozen teacher 的 token distribution。我们要验证：

> 在标准 response-side OPD 之外，对 prompt 内所有 causal prefix positions 进行
> teacher–student distribution matching，能否进一步改善 student 的任务表现？

这里利用的是 teacher 在每个 prompt prefix 上的预测分布，不是一个双向编码器式的
“完整 prompt 表示”。暂称 **prefill-position distillation**。

长 prompt、短 response 只是预先声明的潜在优势场景，不是当前 baseline 的筛选条件。
当前先建立可复现的 general OPD setting，再比较 response-only 与 prefill-aware OPD。

## 2. 当前 baseline authority

当前采用 [THUNLP/OPD](https://github.com/thunlp/OPD) 作为 baseline 实现：

- upstream commit：`4532fd35ccfdde82adc918b265e4c964534e83d1`；
- Student：`Qwen/Qwen3-1.7B-Base`；
- Teacher：`lllyx/Qwen3-4B-Base-GRPO`，frozen；
- train data：upstream `datasets/dapo-math-17k.parquet`；
- eval：AIME24、AIME25、AMC23；
- response-side estimator：upstream `token_reward_direct`；
- generation：non-thinking，temperature `1.0`，每 prompt 4 个 rollouts；
- max prompt / response / model length：1024 / 7168 / 8192；
- global train batch / PPO mini-batch：64 / 64；
- actor learning rate：`1e-6`；
- loss aggregation：`token-mean`；
- top-k：16，`only_stu`，reward weight `student_p`；
- actor、reference、teacher：BF16。

这是当前要先复现的 response-only OPD baseline。它优先于先前使用的
`HJSang/OPSD_OnPolicyDistillation` 路线，也优先于自行构造的 BANKING77 OPD setting。

## 3. Teacher / student qualification

已在 upstream 同类数学 evaluation setting 上做快速 subset qualification，而不是重新跑
整套 paper evaluation：

- tasks：AIME24、AIME25、AMC23；
- 每 task 20 道题，每题 2 rollouts；
- 每个模型共 120 个 rollouts；
- temperature `0.7`，top-p `0.95`，max new tokens `31744`，non-thinking。

结果：

| Model | Accuracy |
|---|---:|
| Qwen3-1.7B-Base student | 4.17% |
| Qwen3-4B-Base-GRPO teacher | 43.33% |
| Teacher - student | +39.17 pp |

paired-problem bootstrap 95% CI 为 `[+28.33, +50.00]` pp；120 个 paired rollout 中，
`teacher correct / student wrong` 为 47，反向为 0。

这足以确认当前模型和 evaluation 能稳定区分 teacher/student，并满足开展 OPD 的前置条件。
它不是对 paper 全量数字的重新复现。

结果文件：
`artifacts/thunlp_opd_reproduction/qualification_n2_60/summary.json`。

## 4. Response-only baseline 的当前工程状态

本地 6 × 48GB GPU 已完成真实三步训练闭环：

- `training/global_step`：1、2、3；
- actor `pg_loss`：0.04813、0.04274、0.04577；
- 三步均完成 backward 和 optimizer update；
- step 3 成功保存 6-rank actor、optimizer 和 extra-state checkpoint；
- 主进程退出码为 0。

产物：

- log：`artifacts/thunlp_opd_reproduction/response_only_paper_short/20260730_213652/train.log`；
- checkpoint：同目录下 `checkpoint/global_step_3/actor`。

这证明 response-only OPD 的 rollout、student/reference/teacher forward、loss、backward、
optimizer 和 checkpoint 路径已经闭环。它只属于工程资格验证，不是 baseline 的科学复现结果。

## 5. 当前硬件决策

正式 baseline 迁移到至少 4 × A100 80GB 的机器，不继续在本地 48GB 卡上缩小正式设置。

原因：upstream 风格的 32768-token per-rank dynamic budget 在本地 48GB 卡上会在
full-vocabulary `log_softmax` 处 OOM；本地闭环需要降到 8192。A100 launcher 恢复 32768，
同时保持 global batch、rollout 数、sequence length 和 optimizer 设置不变。

A100 复现仓库：
[Sisyphe-lee/opd-baseline-repro](https://github.com/Sisyphe-lee/opd-baseline-repro)，
commit `3da0848f446777f0aea055c91b11a4ecb0394b5f`。

该仓库当前为 public，提供：

- 固定 upstream commit；
- 一键环境、权重下载与 runtime qualification；
- 4 × A100 三步 smoke；
- full one-epoch response-only launcher；
- 自动验收 optimizer steps 和 checkpoint shards 的脚本。

## 6. 下一阶段 gate

严格按以下顺序推进：

1. 在 A100 机器运行环境和四卡 NCCL/CUDA Graph qualification；
2. 运行三步 response-only smoke，并由 `verify_run.py` 验收；
3. 运行完整 response-only OPD baseline；
4. 用 AIME24、AIME25、AMC23 评价 initial student、teacher 和 trained student；
5. 只有 baseline 的训练/evaluation 闭环稳定后，才实现 prefill-aware 分支。

此阶段不做 lambda sweep、TIP token selection、hidden-state KD、额外 teacher 或新 benchmark。

## 7. Prefill-aware 方法的最小定义

对于：

```text
[prompt_0 ... prompt_{m-1}, response_0 ... response_{n-1}]
```

causal logits 的监督位置为：

```text
prefill positions:  0 ... m-2
response positions: m-1 ... m+n-2
```

最后一个 prompt position 预测第一个 response token，因此属于 response loss。两个区域必须
分别归一化：

```python
response_loss = masked_mean(token_kd, response_mask)
prefill_loss = masked_mean(token_kd, prefill_mask)
loss = response_loss + lambda_prefill * prefill_loss
```

所以概念上的主要变化确实接近“把 prompt mask 打开”，但不能简单对整个拼接序列做一次
全局平均，否则 prompt 长度会隐式改变 loss 权重。

## 8. 后续最小对照

baseline 稳定后，第一轮只比较：

| 分支 | 目标 |
|---|---|
| Response-only OPD | upstream response loss |
| Prefill-aware OPD | response loss + `lambda * prefill loss` |

两者固定相同 initialization、prompt stream、sampling protocol、global batch、optimizer 和
update budget。先用一个合理的固定 lambda 获取方向性 signal；没有 signal 前不扩建论文级 pipeline。

若得到正 signal，再补 hard prompt CE、compute matching、更多 seeds、长度/context dependency
分桶和自然任务泛化。

## 9. 历史探索的定位

2026-07-29 的 Qwen3-4B/8B + BANKING77 实验曾确认两点：

- response-only OPD 可以非常快地追平 stock teacher；
- uniform prefill KD、`lambda=0.1` 在两个 BANKING77 prompt-length settings 上没有正收益。

这些是有用的负向历史证据，但由于模型、teacher 构造、baseline repo 和 benchmark 都已切换，
不再定义当前 go/no-go。当前首先复现 THUNLP/OPD setting，再在同一 setting 上加入 prefill loss。
