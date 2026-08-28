# PM-A policy census report

Decision: **NOT RUN — PM-A0 workload insufficient**

PM-A was intended to test whether A100 and patched Ascend 910B realize stable,
prompt-dependent behavior policies that alter a canonical A100 GRPO learner
gradient beyond process and resampling noise. The scientific comparison did not
start because its preregistered workload-construction gate failed.

The final A100-only screening completed 256 candidates × 8 samples with no
verifier errors. Exact math, code, and JSON/schema supplied at least 32
nonzero-reward-variance groups, while choice/logic supplied 27. The protocol
required exactly 32 prompts from every family and prohibited replacing the five
missing logic prompts with another family. Two earlier candidate designs and
the final composite-reward design are retained in
`docs/PM_A0_WORKLOAD_REPORT.md`.

Consequences:

- C-P is `not_tested_workload_insufficient`, not passed or failed;
- `U_noise` was not estimated;
- no 910B trajectory, cross-score, gradient, or self-noise result was produced;
- C0 and C1 remain `reopened_not_tested`;
- controller, networking, online LoRA transfer, consistency, and RTX remain out
  of scope;
- no fourth workload-tuning attempt is authorized under this protocol.

The prior vLLM-Ascend issue remains a runtime side-branch matter and did not
cause this stop: all three PM-A0 A100 engine processes and deterministic
verifiers completed normally.

