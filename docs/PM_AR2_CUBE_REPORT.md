# PM-AR2 matched-cube validation report

Decision: **VALID — PM-AR3 unblocked; C-P remains not adjudicated**

PM-AR2 used the frozen 4×27 workload and primary full-support sampler. It did
not alter the predecessor state:

```text
PM-A0 = workload_insufficient_stopped
C-P   = not_tested
```

## Collected evidence

Each backend completed five sequential fresh processes. Every process contains
108 prompt groups and 864 free trajectories, for 4,320 trajectories per
backend and 8,640 in total. Nonzero-advantage group counts were:

| Restart | A100 | patched 910B |
|---:|---:|---:|
| 0 | 93 | 89 |
| 1 | 92 | 93 |
| 2 | 93 | 92 |
| 3 | 85 | 93 |
| 4 | 96 | 90 |

For every restart, the A100 and 910B decode paths each forced-replayed the
union of both free-history sets: 1,728 trajectories per scoring process. The
maximum processed-logprob reporter gaps were:

| Restart | A100 | patched 910B |
|---:|---:|---:|
| 0 | 2.861034318e-6 | 3.695481155e-6 |
| 1 | 2.861034318e-6 | 3.104061761e-6 |
| 2 | 2.861034318e-6 | 3.104061761e-6 |
| 3 | 2.861034318e-6 | 3.104061761e-6 |
| 4 | 2.861034318e-6 | 3.104061761e-6 |

All ten values are below the frozen `5e-5` PM-RQ tolerance. All forced token
sequences, trajectory hashes, prompt/sample identities, finite-value checks,
model/workload identities, seeds, and device schedules passed deterministic
validation.

The machine-readable validation is
`results/pm_ar/PM_AR2_CUBE_VALIDATION.json`. This report establishes only that
the matched cube is valid input to PM-AR3; it does not inspect or adjudicate
the heterogeneous gradient contrast.

## PM-AR3 analysis freeze

Before generating any PM-AR3 gradient, `configs/pm_ar3.json` freezes the four
gradient components, hierarchical bootstrap, 10,000 replicates, seeds,
finite-sample lower-quantile rule, restart halves, prompt-risk statistic, and
the deterministic 4A+4N ESS composition. Its SHA-256 is
`cbfa0bb6983d4766857b7d7a1703ed83b9f68d8a8181ebfef8643185c21fd727`.
