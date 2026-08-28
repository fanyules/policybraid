# Runtime prerequisite

PM-A uses a production-engine probability interface that was qualified before
this scientific gate. The qualification is evidence for measurement validity,
not a PolicyBraid contribution.

The frozen prerequisite is:

- A100 and patched Ascend 910B passed normal-mode probability qualification in
  three fresh processes per platform;
- the tolerance was locked from A100 controls at `5e-5` before reading the NPU
  qualification result;
- A100 produced 1,536 oracle comparisons/error observations and patched 910B
  produced 1,536; all 3,072 comparisons passed;
- all 3,072 forced-replay checks had zero token-logprob replay error;
- the qualification used 49,152 empirical draws and all 16 multinomial cells
  passed their registered tests.

Provenance remains on the side branch:

- experiment adjudication: branch `codex/policymesh-runtime-reentry`, commit
  `9076c23`;
- failure-path hardening: commit `b2b5dc1`;
- isolated vLLM-Ascend runner patch: commit `a94bfcb0` in the compute-host
  vLLM-Ascend worktree.

The earlier failure came from an unsupported standalone/private sampler
invocation and an unpropagated production-runner mode. Its detailed logs,
patch-development history, and old Gate reports are deliberately not present
on `main`; inspect the two archival branches when runtime debugging requires
them. PM-A retains only a small per-process sentinel and never reopens that
engineering investigation.

