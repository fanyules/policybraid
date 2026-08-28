# PM-AR: maximal-balanced scientific re-entry

Status: PM-AR0/PM-AR1 passed; PM-AR2 cube validated; PM-AR3 statistics frozen

Claim under test: C-P

## 1. Relationship to PM-A0

PM-AR is a new, final scientific re-entry Gate. It does not reinterpret or
replace the frozen predecessor state:

```text
PM-A0 = workload_insufficient_stopped
C-P   = not_tested
```

PM-A0 tested a 4×32 workload contract, not the heterogeneous-runtime
hypothesis. PM-AR changes only the common per-family sample count. All model,
runtime, sampling, verifier, reward, gradient, ESS, prompt-risk, and stopping
thresholds remain unchanged.

## 2. PM-AR0: deterministic 4×27 freeze

The only source is revision-3 candidate set
`81261a006addec438ca1677f9fc4f282e99f62c6986d6a1efe7f01961c5007c0`
and its already frozen A100 screening adjudication. No new generation,
resampling, verifier edit, reward edit, parser edit, seed change, or length
change is allowed.

For each of exact math, choice/logic, code unit tests, and JSON/schema/tool:

1. retain records whose existing adjudication has `eligible=true`, a healthy
   verifier, and nonzero group reward variance;
2. sort by the existing `prompt_id`;
3. take the first 27 records;
4. seal every unselected eligible record as non-substitutable surplus.

This produces 108 prompts with equal family weight. Selection may not inspect
reward magnitude, variance magnitude, output text, or any backend metric. The
32 A100 scoring anchors are the first eight selected IDs in each family. PM-AR0
passes only when the selected file, IDs, anchors, 23 surplus IDs, source hashes,
and deterministic reconstruction check are committed and pushed to GitHub
before any PM-AR 910B run.

PM-AR0 establishes workload availability only; C-P remains `not_tested`.

## 3. Frozen execution and learner contract

| Dimension | Value |
|---|---|
| Model | Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Precision / TP | BF16 / TP1 |
| Runtime mode | normal |
| Group | one 8-sample prompt group per engine call |
| Primary sampler | temperature 1, top-p 1, top-k 0, maximum 128 tokens |
| LoRA | rank 8, alpha 8, dropout 0, `q_proj` and `v_proj`, base frozen |
| Objective | clipped GRPO token surrogate, generated-token mask |
| PPO ratio clip | 0.2 |
| Advantage | population-normalized per group, epsilon `1e-4` |
| Reduction | token mean → group mean → prompt mean |

Behavior denominators are detached. Gradient coordinates are sorted by
parameter name and stored in float32. Long importance weights use CPU float64
log space. Distinct execution contexts remain separate strata in
\(\mu_b(a\mid s,c)\).

## 4. PM-AR1: A100-only noise lock

No PM-AR 910B scientific result may be generated or opened before the final
`U_noise` artifact is committed and pushed.

Five A100 restarts run sequentially according to the frozen physical-device
schedule; simultaneous processes are invalid repeats. PM-AR1 measures:

1. **Trainer process/batch noise:** the same histories, rewards, behavior
   logprobs, LoRA initialization, objective, and microbatch contract in five
   fresh learner processes.
2. **A100 rollout-scoring noise:** the same 32 anchor histories cross-scored in
   five fresh A100 engine restarts, followed by the same canonical learner
   gradient calculation on those anchors.
3. **A100 resampling noise:** two disjoint registered seed groups for every one
   of the 108 prompts, followed by canonical learner gradients.

The distance remains

\[
D_g(g_1,g_2)=\frac{\lVert g_1-g_2\rVert_2}
{\lVert g_1\rVert_2+10^{-12}}.
\]

For the five-restart components, compare restart 0 with restarts 1--4 and take
the empirical 0.95 quantile using the conservative `higher` rule. For
resampling, use 10,000 task-stratified prompt bootstraps with seed `24150000`.
Freeze

\[
U_{noise}=\max(U^{trainer}_{95},U^{A100-scoring}_{95},
U^{A100-resample}_{95}).
\]

Failed, OOM, incomplete, device-mismatched, or non-finite runs remain in the
audit trail and cannot be silently replaced.

## 5. PM-AR2: matched A100/910B cube

Only after PM-AR1 is committed, each backend runs:

\[
108\ prompts\times8\ samples\times5\ restarts=4{,}320\ trajectories.
\]

For free histories \(H_A\) and \(H_N\), every action in their union is
cross-scored under the canonical trainer, A100 engine, and patched 910B engine.
Different free histories are never aligned to form a pseudo-KL. Every process
retains PM-RQ sentinels with absolute reporter/oracle tolerance `5e-5`. RTX is
out of scope.

## 6. PM-AR3: gradient adjudication

On the same A100 learner compute

\[
g_{A,A},\quad g_{A,N},\quad g_{N,A},\quad g_{N,N},
\]

with the denominator detached and every objective setting frozen. Report the
actual contrast, trajectory-source component, denominator component, and
interaction residual.

PM-AR passes only if every original scientific threshold holds:

- all PM-RQ sentinels pass;
- the conditional-policy gap exceeds both platforms' self-noise;
- \(L_{95}(D_g)\ge\max(0.05,2U_{noise})\);
- at least four of five restart contrasts exceed the same threshold;
- the 4A+4N mixture ESS-fraction 95% lower bound is at least 0.5;
- at least 64 nonzero-advantage groups cover at least two families;
- prompt risk has the registered cross-process structure; and
- the result holds under primary full-support sampling.

## 7. Fixed decision routing

- A heterogeneous gradient contrast within self-noise stops PolicyBraid.
- A stable global offset without prompt-conditioned structure does not justify
  a prompt-aware controller; only consistency mode or a fixed quota may be
  considered.
- A full PM-AR pass enters PM-B, which must compare same-quota random placement
  with a same-quota risk oracle before any controller implementation.
- If the PM-B oracle does not materially beat random placement, stop the
  controller path.
- RTX and the 1 GbE three-island experiment remain blocked until PM-B passes.
