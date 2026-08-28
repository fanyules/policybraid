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

The engineering diagnosis and patch-development history are deliberately absent
from `main`. A concise description is isolated on public branch
`archive/vllm-ascend-note`; raw diagnostics remain in local archival history.
PM-A would retain only a small per-process sentinel and would not reopen that
engineering investigation.
