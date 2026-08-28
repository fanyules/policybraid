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

## Candidate design revision 2

Revision 2 fixed the output-length and hidden-schema problems, but remained
workload-insufficient:

| Family | Eligible groups | Required |
|---|---:|---:|
| Exact math | 2 | 32 |
| Choice/logic | 13 | 32 |
| Code unit tests | 30 | 32 |
| JSON/schema/tool | 4 | 32 |

The short exact-answer contract made many prompts deterministic across the
eight registered seeds: 61/64 math groups and 50/64 JSON groups had zero
reward, while many others were uniformly correct. Code was close to the
registered target but still failed closed. The immutable evidence is under
`results/pm_a/screening/candidate_set_901e710a/`.

The third and final workload-construction attempt changes the evidence unit,
not the Gate threshold: each prompt contains several deterministic atomic
checks and receives their mean score in `[0,1]`. Exact math and logic use four
independent registered answers; code uses the fraction of hidden unit tests;
JSON uses exact leaf-path values with an extra-field penalty. This corrects the
unnecessary binary-reward restriction introduced by the implementation—the
PM-A protocol only requires deterministic verifiers and nonzero group reward
variance. All task answers, scoring weights, and parsers are frozen before the
third A100 screening. If the third set is insufficient, PM-A0 stops for user
review rather than iterating again.

## Final candidate design revision 3

Revision 3 completed all 2,048 A100 samples with zero verifier errors. Its
registered adjudication was:

| Family | Eligible groups | Required | Result |
|---|---:|---:|---|
| Exact math | 38 | 32 | sufficient |
| Choice/logic | 27 | 32 | insufficient |
| Code unit tests | 33 | 32 | sufficient |
| JSON/schema/tool | 33 | 32 | sufficient |

Because every family was required to contribute 32 prompts, the five-prompt
logic shortfall cannot be filled with surplus prompts from another family.
Revision 3 is consequently `WORKLOAD_INSUFFICIENT`, no selected 128-prompt file
was produced, and the registered no-fourth-attempt rule closes PM-A0. The raw
evidence is under `results/pm_a/screening/candidate_set_81261a00/`.

This outcome does not adjudicate C-P. PM-A1--PM-A3 and all 910B scientific runs
were never started.
