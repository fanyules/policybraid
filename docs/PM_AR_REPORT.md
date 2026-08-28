# PM-AR final report

Decision: **FAIL — stop PolicyBraid**

This result does not reinterpret the predecessor workload gate:

```text
PM-A0 = workload_insufficient_stopped
C-P before PM-AR = not_tested
```

PM-AR was the independent maximal-balanced scientific re-entry. Its 4×27
workload, PM-AR1 noise lock, PM-AR2 matched cube, PM-AR3 statistics, and all
thresholds remained frozen. PM-AR now adjudicates C-P as failed under the
registered primary configuration.

## Gate result

| Mandatory criterion | Observed | Threshold | Result |
|---|---:|---:|---|
| PM-RQ reporter/oracle error | max 3.6955e-6 | ≤5e-5 | pass |
| Conditional A100↔910B logprob gap | p95 0.04134 | above both self-error floors | pass |
| Gradient distance lower 95% bound | 0.18678 | ≥1.13998 | **fail** |
| Restart replication | 0/5 | ≥4/5 above 1.13998 | **fail** |
| 4A+4N ESS-fraction lower 95% bound | 0.90442 | ≥0.5 | pass |
| Nonzero-advantage support | minimum 85, all 4 families | ≥64, ≥2 families | pass |
| Prompt-risk structure | Spearman L95 0.97457; 4 families directional | L95 >0.3; ≥2 families | pass |
| Primary configuration | full support, normal mode | required | pass |

The five restart gradient distances were `0.54481`, `0.45157`, `0.24138`,
`0.40668`, and `0.42632`. The aggregate distance was `0.27820`, while the
pre-registered boundary was

```text
max(0.05, 2 × U_noise)
= max(0.05, 2 × 0.5699915723)
= 1.1399831447.
```

The bootstrap used 10,000 paired hierarchical resamples with the frozen seed
and produced p05/p50/p95 distances of `0.18678/0.31985/0.57564`. The upper
tail still remained below the registered boundary. The negative decision is
therefore not caused by a correctness failure, unusable importance weights,
insufficient reward variation, or unstable prompt risk. The heterogeneous
runtimes do exhibit measurable and highly repeatable prompt-conditioned
logprob differences, but the resulting canonical learner-gradient contrast
does not exceed the pre-registered A100 resampling-noise boundary.

## Gradient decomposition

Relative to the norm of `g_A_A`:

| Component | Normalized norm |
|---|---:|
| Actual `g_N_N - g_A_A` | 0.27820 |
| Trajectory source `g_N_A - g_A_A` | 0.35015 |
| Denominator `g_A_N - g_A_A` | 0.49378 |
| Interaction residual | 0.47281 |

The aggregate `g_A_A`/`g_N_N` cosine was `0.98185`. These diagnostics are
reported as mechanism evidence only; they cannot rescue the failed mandatory
gradient thresholds.

## Evidence and stopping action

The machine-readable decision is
`results/pm_ar/PM_AR_ADJUDICATION.json`. Five learner manifests and five
canonical trainer-score files are under `results/pm_ar/pm_ar3/`. The five raw
per-prompt gradient tensors remain on the A100 evidence host at
`/data/policybraid-artifacts/pm_ar3/gradients/`; their SHA-256 identities are
recorded in the adjudication and manifests.

Under the frozen routing rule:

- do not enter PM-B;
- do not implement a PolicyBraid controller;
- do not add RTX or the 1 GbE three-island experiment;
- do not lower or replace `U_noise`, reinterpret the 0.278 aggregate distance,
  or tune the sampler/workload after observing this result.
