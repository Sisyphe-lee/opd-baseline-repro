# Two-stage distillation zero diagnostic

## Question and claim boundary

This experiment tests whether TCOD's early-horizon benefit is mainly cold-start
compensation: first move the base student toward successful teacher behavior,
then run otherwise full-horizon Vanilla OPD from that initialization.

Stage 1 is **offline distillation**, implemented as teacher-success hard-label SFT / sequence KD, not soft-logit KD.
For accumulated history $h$ and a sampled teacher response $y$ it minimizes

$$
\mathcal L_{\mathrm{SFT}}
=-\mathbb E_{(h,y)\sim D_T^+}
\left[\frac{1}{|y|}\sum_{j=1}^{|y|}
\log \pi_S(y_j\mid h,y_{<j})\right].
$$

Without success filtering, Monte Carlo teacher samples estimate the cross
entropy $H(\pi_T,\pi_S)=H(\pi_T)+D_{\mathrm{KL}}(\pi_T\|\pi_S)$ on teacher
states. Here $D_T^+$ is additionally conditioned on successful trajectories,
so it is narrower than the teacher's full state distribution. It must not be
described as reverse-KL offline distillation.

Stage 2 samples states from the warm-started student and applies the frozen
OPD loss contract. It therefore covers student-induced and recovery states that
the successful offline set cannot cover.

Teacher rollout collection and JSONL construction are preprocessing, not additional learning stages.

The two-stage schedule alone is a diagnostic baseline, not a novelty claim.

## Frozen zero-test design

| Component | Setting |
|---|---|
| Teacher collection | GiGPO Qwen2.5-7B, all 3,553 train games, one rollout/game, seed 42 |
| Collection contract | corrected step-2 prompt, accumulated memory, strict lowercase action parser, horizon 30, response 512, temperature 0.4 |
| Accepted trajectories | task success and a strict parsable response at every recorded turn |
| Offline sample unit | one accumulated conversation prefix and one final teacher response per row |
| Offline budget (SFT/SeqKD) | 30 updates, global batch 64, LR $10^{-6}$ |
| Online budget | 220 Vanilla OPD updates, batch 16/train batch 64, LR $10^{-6}$, horizon 30 |
| Total | 250 optimizer updates |
| Evaluation | frozen full274 seed 42 contract |

The first zero diagnostic uses
$N_{\mathrm{off}}:N_{\mathrm{on}}=30:220=3:22$ (12% versus 88% by
update count, not by compute). The online job saves full and Hugging Face
weights at every 20 updates, including step 220.


Every prefix is checked using the student tokenizer. Rows with prompt length
above 10,240 or response length above 512 are excluded and counted in the data
manifest. By default, a successful trajectory may contain an earlier
inadmissible recovery action; those rows are retained and counted. The stricter
`--require-all-admissible` dataset is an ablation, not the default, because
silently selecting only flawless trajectories would further narrow state
coverage.

The SFT formatter uses `enable_concatenated_multi_turn: false`: earlier turns
are context and only the final assistant response in each row bears loss. This
matches the per-turn unit consumed by online OPD and avoids training the same
assistant token multiple times within one row.

## Run sequence

All GPU jobs are tmux-only and refuse to mix with existing outputs.

1. The all-in-one driver collects teacher records on eight GPUs and builds the
   token-validated offline dataset when those artifacts do not already exist.

2. Inspect the generated manifest before training:

   `data/two_stage_distillation/generated/teacher_success_prefix_seqkd_seed42.manifest.json`

   At minimum verify record coverage is 3,553, `output_samples >= 1920`, the
   rejected-success examples, over-limit count, and inadmissible-row count.

3. Run the complete zero diagnostic on all eight GPUs:

   ```bash
   scripts/launch_two_stage_zero_diagnostic_tmux.sh 0,1,2,3,4,5,6,7
   ```

   The tmux driver evaluates the identically configured student initialization,
   trains offline 30, evaluates offline 30, generates the warm-start figures,
   trains Vanilla OPD 220 from the offline Hugging Face weights with a fresh
   optimizer, evaluates the final policy, and regenerates comparison plus
   entropy/KL figures. The explicit init evaluation is needed because an
   offline checkpoint cannot be called "warm" from teacher-forced loss alone.

Outputs are rooted at `runs/experiments/two_stage_distillation/`; synthesized
tables and figures are under
`analysis/two_stage_zero_diagnostic_offline30_online220_seed42/`.

## Diagnostics and cost accounting

The offline-distillation stage has a pre-declared diagnostics exemption: it has no
student rollout or teacher-scoring process, so per-turn teacher/student entropy,
surprisal, and state-outcome plots do not exist for that stage. Its dataset
manifest, loss logs, exact config, checkpoint, and W&B/TensorBoard assets remain
mandatory.

The online stage has no exemption. It uses
`OPD_entropy_mask_promptfix_alfworld_workflow` with
`entropy_frontier_strategy: full`, which returns every real response to loss
while retaining the required trajectory/token diagnostics. This is Vanilla OPD
at the policy/loss-selection level, but the extra top-16 instrumentation changes
wall time. A compute claim therefore requires a base-initialized full-loss arm
with the same instrumentation and GPU layout.

The equality $30+220=250$ matches update count only. Teacher trajectory
generation is extra acquisition cost, and SFT/OPD updates process different
token volumes. Report at least:

- teacher-generated response tokens and environment steps;
- offline-stage trainable tokens and GPU-hours;
- online student generation, teacher-scored, and trainable tokens;
- end-to-end GPU-hours and wall time.

Do not claim equal compute from equal updates.

## Required interpretation matrix

The decisive comparison is not just the final hybrid versus frozen TCOD:

| Arm | Purpose |
|---|---|
| Base $\rightarrow$ Vanilla OPD | cold-start control |
| Base $\rightarrow$ TCOD | frozen curriculum reference |
| Offline only (teacher-success SeqKD) | how much behavior cloning itself contributes |
| Offline $\rightarrow$ Vanilla OPD | primary zero diagnostic |
| Offline $\rightarrow$ TCOD | whether horizon control still helps after warm-start |

If offline-to-Vanilla closes the TCOD gap, the parsimonious explanation is that
much of TCOD's gain is cold-start handling. If offline-to-TCOD still beats both,
horizon/state heterogeneity remains useful after basic competence. If offline-only
looks good but degrades during OPD, investigate objective/distribution shock
instead of concluding that online distillation is intrinsically harmful.

Relevant implementation files are the three `two_stage_*.yaml` experiment
configs, `build_teacher_success_sft.py`, and the two tmux launchers. Generated
records, datasets, checkpoints, logs, diagnostics, and evaluation results stay
under this repository but are intentionally not committed.
