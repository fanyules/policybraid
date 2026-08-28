# PM-AR2 decode-context cross-score qualification

Status: **PASS on A100 and patched 910B**

## Why the first scorer is not used

PM-AR1 measured a maximum `1.4769647121` token-logprob difference between
decode-time rollout reporting and teacher-forced prompt scoring. These are
different execution contexts, so the faster prefill path cannot serve as the
cross-backend behavior denominator.

## Qualified mechanism

The replacement uses the existing vLLM logits-processor interface. At each
decode step it:

1. observes the full-support logits before modifying them;
2. records the target token logprob via float32 log-sum-exp; and
3. forces the registered target token so the next step sees the exact history.

It changes no model, sampler kernel, top-k/top-p kernel, runtime patch, or
probability normalization. Cross-scoring is instrumented evidence, not a timing
run. vLLM selects its V1 model runner when custom logits processors are active;
this context is explicitly recorded rather than pooled with normal V2 rollout.

## A100 self-replay result

On all 864 seed-group-0 trajectories:

- every forced sequence exactly matched its target token IDs;
- all captured logprobs were finite and length-aligned;
- maximum absolute error against the normal decode reporter was
  `2.8610343179e-6`;
- the registered tolerance is `5e-5`.

The A100 path therefore passes. The raw qualification is
`results/pm_ar/cross_score_preflight/a100_s0_self.json`, SHA-256
`16e3d2ef4ab3693720c3316fed4acd2c05abe0339e1ba1c381dccea93e873073`.

## Patched 910B self-replay result

On formal restart 0 and all 864 trajectories:

- every forced sequence exactly matched its target token IDs;
- all captured logprobs were finite and length-aligned;
- maximum absolute error against the normal 910B decode reporter was
  `3.6954811549e-6`;
- the active runner resolved to isolated patch commit
  `a94bfcb0f4326c443243800111452f496d517c87`;
- the top-level mode was `processed_logprobs`.

The NPU path therefore passes the same `5e-5` qualification. The raw result is
`results/pm_ar/cross_score_preflight/910b_r0_self.json`, SHA-256
`2c59447b2f011b5468c1de29291930f4755d7f2d764af6ff7d15efbcfb62aeb7`.

Two pre-generation audit-import failures remain under
`results/pm_ar/cube/attempts/`. They were resolved by auditing already loaded
runner modules after normal engine initialization; neither attempt produced a
trajectory or occupied an NPU after exit.
