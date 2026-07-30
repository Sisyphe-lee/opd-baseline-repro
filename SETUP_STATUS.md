# THUNLP/OPD baseline setup status

更新时间：2026-07-30

本文件记录当前可继续执行的基础设施状态。研究决策以 `BASELINE_DECISIONS.md` 为准。

## 当前结论

- Teacher/student gap qualification：已完成。
- 本地 response-only OPD 三步闭环：已完成。
- 可迁移的 public reproduction repo：已完成。
- 4 × A100 80GB smoke：待在新机器执行。
- full response-only baseline：尚未启动。
- prefill-aware 改动：尚未进入当前实现阶段。

## 当前代码和模型

- THUNLP/OPD：`third_party/THUNLP_OPD`
- remote：`https://github.com/thunlp/OPD.git`
- pinned commit：`4532fd35ccfdde82adc918b265e4c964534e83d1`
- Student：`/data1/lcy/cache/modelscope/models/Qwen--Qwen3-1.7B-Base`
- Teacher：`/data1/lcy/cache/huggingface/models/lllyx--Qwen3-4B-Base-GRPO`
- Train data：`third_party/THUNLP_OPD/datasets/dapo-math-17k.parquet`
- Eval data：upstream AIME24、AIME25、AMC23。

upstream checkout 本身无 tracked modifications；`outputs/` 和 Python cache 是未跟踪运行产物。

## 当前 Python 环境

正式环境：`/data1/lcy/projects/opd_prefill/.venv_opd_official`

- Python `3.12.13`
- PyTorch `2.8.0+cu128`
- vLLM `0.11.0`
- Transformers `4.57.6`
- FlashAttention `2.8.1`
- FlashInfer `0.3.1`
- Ray `2.56.1`
- NumPy `1.26.4`
- SciPy `1.15.3`
- Datasets `3.6.0`
- TensorDict `0.10.0`
- CuPy `13.6.0`

`pip check`、真实 BF16 FlashAttention kernel、6-GPU NCCL all-reduce 和 CUDA Graph 均已通过。

旧环境 `.venv`、OPSD checkout、Qwen3-4B/8B、RULER 和 LongICLBench 产物仍保留，但只服务于
历史实验，不是当前 baseline 的执行环境。

## Teacher gap qualification

产物：`artifacts/thunlp_opd_reproduction/qualification_n2_60/summary.json`

- 60 道题、每题 2 rollouts，每模型 120 rollouts；
- student accuracy：4.17%；
- teacher accuracy：43.33%；
- gap：+39.17 pp；
- paired-problem bootstrap 95% CI：`[+28.33, +50.00]` pp。

该 subset 已足够确认模型/evaluation setting 可用于 OPD，不需要在训练前再做一次全量 evaluation。

## 本地三步训练闭环

成功 run：
`artifacts/thunlp_opd_reproduction/response_only_paper_short/20260730_213652`

- 6 × 48GB GPU；
- global train batch 64、4 rollouts、prompt 1024、response 7168；
- BF16；
- `MAX_TOKENS_PER_GPU=8192`；
- optimizer steps 1、2、3 完成；
- step 3 的 6-rank checkpoint 完整；
- 主进程 exit code 0。

每步 wall time 约为 116.2s、104.5s、103.9s；第三步另用约 6.9s 保存 checkpoint。

已知但不构成失败的现象：训练和 checkpoint 完成后，Ray teardown 期间可能打印
DataLoader worker 被 `SIGKILL` 的 traceback。验收以主进程 exit code、三步 metrics、成功标记和
完整 checkpoint shards 为准。

## 已确认的本地约束

1. 32768-token per-rank budget 会在 full-vocabulary `log_softmax` 处触发 48GB OOM；本地 smoke
   使用 8192。A100 正式 launcher 恢复 32768。
2. `trainer.rollout_data_dir` 必须保持 `null`。upstream 当前 rollout dump 会尝试把 Tensor 直接
   `json.dumps`，在第一次 optimizer update 后报错；baseline 不需要该 dump，因此没有修改 upstream。
3. 保留 CUDA Graph；当前 NCCL + CUDA Graph qualification 已通过。

## Public A100 reproduction repo

- URL：`https://github.com/Sisyphe-lee/opd-baseline-repro`
- visibility：public
- commit：`3da0848f446777f0aea055c91b11a4ecb0394b5f`
- target：4 × A100 80GB

新机器执行：

```bash
git clone https://github.com/Sisyphe-lee/opd-baseline-repro.git
cd opd-baseline-repro

CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/prepare_a100.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/run_smoke_4xa100.sh
./scripts/verify_run.py
```

三步验收通过后再启动 full baseline：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/run_full_4xa100.sh
```

`prepare_a100.sh` 会固定 upstream commit、创建隔离环境、下载两个模型，并验证
FlashAttention、四卡 NCCL 和 CUDA Graph。脚本支持重复运行和断点复用。

## 当前 handoff

下一 session 不需要继续修改本地 infra，也不需要先实现 prefill loss。唯一主线是：

1. 在 A100 机器运行 `prepare_a100.sh`；
2. 运行并验收三步 smoke；
3. 若通过，启动 full response-only OPD；
4. full run 完成后执行 upstream benchmark evaluation；
5. baseline 结果稳定后，再实现和比较 prefill-aware OPD。
