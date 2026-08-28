# PM-AR1 A100 noise-lock report

Decision: **PASS — `U_noise` locked; PM-AR2 unblocked**

PM-AR1 ran only on A100. No 910B scientific trajectory or score was generated
before this lock.

## Inputs and execution

- selected workload: 108 prompts, 27 per family;
- two disjoint A100 seed groups: 864 trajectories each;
- nonzero-advantage groups: 87/108 in both seed groups;
- trainer process repeats: five sequential fresh processes on GPU
  0→1→2→3→0;
- A100 scorer repeats: five sequential fresh engine processes on the same device
  schedule and the fixed 32-anchor set;
- learner: Qwen3-1.7B BF16 with 56 wrapped q/v projections and 1,605,632
  trainable rank-8 LoRA parameters;
- full-workload learner peak allocation: 6.77 GB;
- no OOM, non-finite gradient, missing ownership, or incomplete formal run.

One engineering attempt failed before generation because Transformers 5.5
returned a mapping from the chat template. The incomplete artifact is retained
under `results/pm_ar/noise/attempts/`; a tested input-normalization fix preceded
both formal seed groups.

## Locked components

| Noise source | Evidence | 95% upper bound |
|---|---|---:|
| Trainer process/batch | r0 against r1–r4, same histories | 0.0014447144 |
| A100 scoring restart | r0 against r1–r4, same anchors and scorer context | 0.0013993299 |
| A100 resampling | 10,000 four-family paired prompt bootstraps | 0.5699915723 |

Therefore:

\[
U_{noise}=0.5699915723,
\qquad
\max(0.05,2U_{noise})=1.1399831447.
\]

The full two-seed aggregate gradient distance is `0.2023672709`. Its bootstrap
distribution has median `0.3554701377`, p90 `0.5077834196`, p95
`0.5699915723`, and p99 `0.7610161647`. Resampling—not learner-process or
scorer-restart instability—dominates the registered noise floor. This high
threshold is retained unchanged for PM-AR3.

## Context boundary

Teacher-forced prompt scoring and decode-time rollout reporting are distinct
execution contexts. Their largest token-logprob gap was `1.4769647121`; PM-AR1
does not average or reinterpret this systematic context contrast as restart
noise. The scorer component compares r0–r4 only within the identical
teacher-forced context. PM-AR2 must continue to report cross-score context
separately from the actual rollout behavior denominator.

## Artifact policy

The two 668 MB per-prompt gradient tensors and other raw gradient tensors remain
on the A100 host. Their SHA-256 identities are recorded in
`results/pm_ar/PM_AR1_NOISE_LOCK.json` and the committed manifests. JSON
trajectories, scorer outputs, manifests, the incomplete preflight, and the
compact noise lock are retained in the repository.

