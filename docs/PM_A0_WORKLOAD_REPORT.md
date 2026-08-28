# PM-A0 workload construction report

Status: `WORKLOAD_INSUFFICIENT`  
Candidate-set SHA-256: `856df3c82abaac6d09d5481606b546151a147d564d9bbcdcc000679457824a7c`  
A100 screening process: complete; 256 candidates × 8 samples  
Effect on C-P: none; the scientific claim remains untested

## Adjudication

The first candidate set did not provide 32 nonzero-reward-variance groups per
family and therefore cannot enter PM-A1.

| Family | Eligible groups | Required |
|---|---:|---:|
| Exact math | 13 | 32 |
| Choice/logic | 0 | 32 |
| Code unit tests | 14 | 32 |
| JSON/schema/tool | 0 | 32 |

All 2,048 verifier invocations completed. Selection used no 910B result and no
backend difference.

## Failure diagnosis

This was a workload-contract failure rather than a model or runtime failure.

- Choice/logic produced 128-token explanations without reaching the registered
  final-answer marker in every sample.
- Exact-math recurrence, combinatorics, and congruence prompts frequently hit
  the same output-length limit before the final marker.
- JSON prompts did not state their exact nesting/key skeleton strongly enough;
  outputs were often semantically plausible but disagreed with the hidden
  structural contract.
- Code tasks separated into nearly deterministic easy and hard buckets, leaving
  too few medium-difficulty groups.

## Registered revision boundary

The next candidate set may use only these A100-only observations. It will:

1. require answer-only output for math and choice tasks;
2. expose the exact JSON key/nesting contract while retaining hidden values;
3. replace deterministic code buckets with task forms whose first-set A100
   accuracy was neither zero nor one; and
4. retain 64 candidates per family, eight samples, the original independent
   screening seed, fixed-ID selection, and all PM-A scientific thresholds.

The original candidate file and raw screening remain immutable under
`results/pm_a/screening/candidate_set_856df3c8/`. Screening trajectories remain
excluded. No PM-A1 or 910B scientific run was started.

