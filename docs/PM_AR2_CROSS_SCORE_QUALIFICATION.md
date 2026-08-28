# PM-AR2 decode-context cross-score qualification

Status: **A100 PASS; patched 910B qualification pending**

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

Patched 910B must independently satisfy the same forced-token, finite-value,
length, and `5e-5` self-reporter checks before any cross-backend score is used.
Failure leaves PM-AR2 incomplete and does not become evidence about C-P.

