# Archived vLLM-Ascend runtime note

This branch records one engineering prerequisite that preceded PolicyBraid. It
is not a scientific result or a paper contribution.

An earlier test invoked an internal Ascend sampler outside its production model
runner and also found that the production runner did not propagate the selected
logprob mode to the sampler. The standalone call violated a runner-private input
contract; it was removed from research gating. A caller-level mode-propagation
fix was then tested in an isolated vLLM-Ascend worktree. It did not change a
sampling kernel, renormalize outputs, or modify the model.

The resulting production interface passed the registered A100 and patched 910B
probability qualification with the A100-locked `5e-5` tolerance. This qualified
interface was treated only as a measurement prerequisite for PM-A.

Raw failure logs, patch-development attempts, and large runtime captures are
intentionally absent from the public repository. They remain in the local
archival development history for runtime debugging. PolicyBraid `main` contains
only the compact prerequisite statement and the scientific workload evidence.

