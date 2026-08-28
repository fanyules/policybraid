# PM-A: heterogeneous behavior-policy census

Status: preregistered; PM-A0 candidate design revision 2 frozen  
Claim adjudicated: C-P only  
Platforms in the primary gate: A100 and patched Ascend 910B

## 1. Scientific question and estimand

At a fixed checkpoint, matched prompt/group workload, and explicitly recorded
execution context, do A100 and patched 910B realize stable prompt-dependent
behavior policies that alter the canonical GRPO learner gradient beyond
matched execution and resampling noise?

The behavior policy is conditional on runtime context:

\[
\mu_b(a\mid s,c),
\]

where `c` includes batch composition, batch/bucket size, request order,
graph/eager path, TP degree, runtime/version, and packing schedule. Distinct
contexts are separate strata. They must not be averaged and reported as a
backend effect.

PM-A adjudicates only:

> **C-P.** At a fixed checkpoint, heterogeneous rollout runtimes realize
> prompt-dependent behavior-policy differences that alter the canonical
> learner gradient beyond matched execution and resampling noise.

C0 (scheduling causality) and C1 (controller benefit) remain untested even if
PM-A passes. This gate contains no controller, online weight transfer, first-K
networking, consistency-mode rescue, or RTX experiment.

## 2. Frozen primary configuration

| Dimension | Frozen value |
|---|---|
| Model | Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Precision / TP | BF16 / TP1 |
| Runtime mode | normal |
| Group | 8 samples from one prompt in one engine call |
| Continuation | at most 128 tokens |
| Primary sampling | temperature 1.0, top-p 1.0, top-k 0 |
| Secondary stress only | temperature 1.3, top-p 0.95, top-k 0 |
| Prompt order | prompt ID, then sample ID |
| Packing | disabled |
| Restarts | five sequential fresh processes per platform |

The full-support primary configuration alone can establish a pass. The stress
configuration cannot rescue a primary negative result. Every host re-verifies
the model manifest; the verification identity, runtime commit, patch commit,
physical device, and realized engine context are written into every process
result.

The learner uses a frozen LoRA coordinate system from the start: rank 8, alpha
8, dropout 0, `q_proj` and `v_proj`, no bias, base weights frozen. Behavior
logprob denominators are detached. Parameter names are sorted lexicographically
before vectors are flattened.

## 3. PM-A0: workload construction and blind screening

The formal workload contains 128 prompts: 32 each from exact mathematics,
choice/logic, code with deterministic unit tests, and JSON/schema/tool
constraints. The repository deterministically builds 64 candidates per family.

Candidate design revision 1 was adjudicated workload-insufficient before PM-A1
and is retained in `docs/PM_A0_WORKLOAD_REPORT.md`. Revision 2 is limited to the
registered answer-contract and difficulty repairs in that report; it does not
change sampling, selection, learner, statistic, pass threshold, or platform
scope.

Candidate screening is performed once on A100 with the independent registered
seed. Each candidate receives one group of eight samples. Selection may use
only:

1. whether the candidate verifier itself completed normally; and
2. whether the eight binary rewards have nonzero variance.

Within each family, the first 32 eligible candidates in fixed prompt-ID order
are selected. Screening trajectories are excluded from every PM-A estimate.
Neither 910B output nor any A100/910B difference may be read or used during
selection. Candidate definitions, raw screening output, selection adjudication,
and selected IDs are hashed and retained.

If any family has fewer than 32 eligible candidates, PM-A0 is `workload
insufficient`; revise the workload under a new protocol commit before any PM-A1
or 910B scientific run. In the formal cube, fewer than 64 nonzero-advantage
groups or fewer than two represented families is likewise workload
insufficient, not evidence for a null backend effect.

## 4. PM-A1: A100-only noise lock

PM-A1 is completed and committed before any PM-A 910B scientific result is
opened. Processes run sequentially and record the physical GPU; simultaneous
PM-RQ processes are not scientific repeats.

Three noise components are frozen:

1. **Trainer process/batch noise.** Recompute the same histories in five fresh
   A100 learner processes.
2. **A100 rollout-scoring noise.** Cross-score 32 fixed anchor prompts in five
   sequential A100 engine restarts under the primary context.
3. **A100 resampling noise.** For every formal prompt, generate two disjoint
   A100 seed groups and compute the corresponding GRPO gradients.

The primary normalized distance is

\[
D_g(g_1,g_2)=\frac{\lVert g_1-g_2\rVert_2}
{\lVert g_1\rVert_2+\epsilon}.
\]

The numerical `epsilon` is fixed by the implementation before observing a
gradient comparison and is reported. Let the task-stratified prompt-bootstrap
upper 95% limits be \(U^{trainer}_{95}\), \(U^{A100-process}_{95}\), and
\(U^{A100-resample}_{95}\). Freeze

\[
U_{noise}=\max(U^{trainer}_{95},U^{A100-process}_{95},
U^{A100-resample}_{95}).
\]

910B self-noise is subsequently measured with the same code, but cannot alter
this statistic, threshold, bootstrap hierarchy, or the 5% practical floor.

## 5. PM-A2: matched trajectory cube

Each platform and restart generates 128 prompts × 8 samples = 1,024
trajectories. Five restarts yield 5,120 trajectories per platform. A trajectory
record contains at least:

| Field | Meaning |
|---|---|
| `prompt_id`, `task_family`, `group_id`, `sample_id`, `seed_id` | registered workload identity |
| `source_backend`, `physical_device`, `restart_id` | source and independent repeat |
| `execution_context` | batch composition/bucket/order, graph path, TP, versions, packing |
| `policy_version`, `model_manifest_sha256` | fixed policy identity |
| `token_ids`, `reward` | realized trajectory and deterministic outcome |
| `processed_behavior_logprobs` | sampling-distribution token logprobs |
| `runtime_commit`, `runtime_patch_commit` | executable provenance |

Counter-based seeds provide stable record IDs only. CUDA and CANN are not
required to consume random numbers identically or generate identical tokens.

Let the free histories be \(H_A\) and \(H_N\). Every realized token/action in
their union is forced-replayed and cross-scored under all three paths:

\[
\log\pi_{train}(a_t\mid h_t),\quad
\log\mu_A(a_t\mid h_t,c_A),\quad
\log\mu_N(a_t\mid h_t,c_N).
\]

KL-like quantities are never computed by aligning unrelated free histories.
Each formal process also runs a small fixed PM-RQ sentinel. A process is invalid
if any reporter/oracle absolute error exceeds `5e-5`.

## 6. PM-A3: canonical-learner gradient decomposition

All gradients are evaluated on the same A100 learner, checkpoint, LoRA
coordinates, objective implementation, and numerical settings. Reward,
group-advantage normalization, PPO/GRPO clipping, token mask, length
normalization, and learning rate are frozen before the cube is scored.

For source \(s\in\{A,N\}\) and detached behavior denominator
\(d\in\{A,N\}\), compute

\[
g_{A,A},\quad g_{A,N},\quad g_{N,A},\quad g_{N,N}.
\]

The actual heterogeneous contrast is

\[
\Delta_{actual}=g_{N,N}-g_{A,A}.
\]

Using \(g_{A,A}\) as the origin, report the trajectory-source component
\(g_{N,A}-g_{A,A}\), denominator component \(g_{A,N}-g_{A,A}\), and interaction
residual

\[
g_{N,N}-g_{N,A}-g_{A,N}+g_{A,A}.
\]

This separates changed answers, changed probabilities on the same actions, and
their interaction.

### Required evidence columns

| Evidence unit | Required measures |
|---|---|
| Token/history | trainer↔A100, trainer↔910B, and 910B↔A100 logprob gaps; support violation |
| Sequence/group | log importance weight, clipping rate, ESS fraction, reward, length |
| Prompt/restart | normalized gradient L2, gradient cosine, nonzero-advantage coverage |
| Learner step | fixed-step held-out policy KL and gradient norm |
| Runtime | passport/cross-score time and fraction, realized context, PM-RQ sentinel |

Long sequence and group weights use CPU float64 log-space accumulation and
log-sum-exp; probabilities are never multiplied directly.

The statistical unit is the prompt. Confidence intervals use paired,
task-stratified bootstrap respecting restart → device → prompt hierarchy.
Tokens are not independent replicates. All interval methods, seeds, and
replicate counts are written to the adjudication artifact before use.

## 7. Prompt-dependent structure

For each prompt,

\[
R_p=p95_t\left|\log\mu_N(a_t\mid h_t)-
\log\mu_A(a_t\mid h_t)\right|.
\]

A controller-relevant structure requires all of:

- prompt-risk IQR greater than twice within-process MAD;
- prompt-risk Spearman correlation across two registered restart halves with a
  95% confidence lower bound above 0.3; and
- the same directional relationship in at least two task families.

A near-global fixed offset can motivate a static quota but fails C-P's
prompt-dependent condition.

## 8. Adjudication

PM-A passes only if the primary configuration satisfies every row:

| Criterion | Registered threshold |
|---|---|
| Measurement validity | every PM-RQ sentinel passes |
| Conditional-policy gap | exceeds A100 and 910B self-noise |
| Actual gradient gap | lower 95% CI of \(D_g\) ≥ `max(0.05, 2*U_noise)` |
| Restart replication | at least 4 of 5 restart contrasts exceed the same threshold |
| Usable mixed behavior | 4A+4N median group-ESS-fraction lower 95% CI ≥ 0.5 |
| Workload support | at least 64 nonzero-advantage groups from at least two families |
| Prompt structure | all Section 7 conditions hold |
| Configuration | result holds under primary full-support sampling |

Stop PolicyBraid if the primary contrast is within process/resampling noise,
the gradient-distance lower bound is below 5%, every practical mixture has ESS
below 0.5, only the stress sampler is positive, prompt risk lacks stable
structure, or the effect is only a global offset.

If PM-A passes, C-P becomes `passed`; C0 and C1 remain
`reopened_not_tested`. Only then may a narrow consistency-mode qualification
and 64-prompt/three-restart control run. If consistency mode removes the
increment at no more than 10% throughput cost, stop. RTX is considered only
after that control, followed by PM-B. No online controller is implemented
unless a same-quota risk-aware oracle in PM-B materially beats same-quota
random placement.

## 9. Immutable artifacts and run order

1. `workloads/pm_a_candidates.jsonl`
2. `results/pm_a/screening/a100_screening.json`
3. `results/pm_a/screening/PM_A0_ADJUDICATION.json`
4. `workloads/pm_a_selected.jsonl`
5. five A100 learner/scoring records and two A100 resample cubes
6. `results/pm_a/PM_A1_NOISE_LOCK.json`
7. five A100 and then five 910B formal process records
8. union-history score records and four gradient components
9. `results/pm_a/PM_A_ADJUDICATION.json`
10. `docs/PM_POLICY_CENSUS_REPORT.md`

Existing artifacts are never overwritten. A rerun uses a new registered run
ID and remains visible. The order above is a scientific control: step 7 cannot
begin on 910B until step 6 is committed.
