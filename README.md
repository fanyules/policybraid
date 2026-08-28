# PolicyBraid

PolicyBraid asks whether heterogeneous rollout runtimes implement stable,
prompt-dependent behavior policies that measurably change the canonical GRPO
learner update at a fixed checkpoint.

PM-A0 remains stopped at its A100-only workload-construction prerequisite. In the final
registered candidate set, exact math (38), code (33), and JSON/schema (33) met
the requirement of 32 nonzero-variance groups, but choice/logic reached only
27. The workload is therefore insufficient. No A100 noise lock, 910B
scientific census, controller, online LoRA transfer, first-K networking,
consistency mode, or RTX experiment was started under PM-A.

The active re-entry Gate is **PM-AR**. It deterministically takes the first 27
eligible revision-3 prompts per family (108 total), preserves equal family
weight, and leaves all scientific thresholds unchanged. C-P remains untested
until PM-AR3.

- Protocol: `docs/PM_POLICY_CENSUS_PROTOCOL.md`
- Frozen configuration: `configs/pm_a.json`
- Runtime prerequisite note: `docs/RUNTIME_PREREQUISITE.md`
- Candidate workload generator: `scripts/build_pm_a_candidates.py`
- Screening adjudicator: `scripts/screen_pm_a_prompts.py`
- Workload outcome: `docs/PM_A0_WORKLOAD_REPORT.md`
- Gate report: `docs/PM_POLICY_CENSUS_REPORT.md`
- PM-AR protocol: `docs/PM_AR_PROTOCOL.md`
- PM-AR configuration: `configs/pm_ar.json`

The former vLLM-Ascend investigation is intentionally absent from `main`.
A concise public description is isolated on branch
`archive/vllm-ascend-note`; raw diagnostics remain only in local archival
history. `main` carries only the qualified interface prerequisite.

## Local checks

```bash
python -m unittest discover -s tests -v
python scripts/build_pm_a_candidates.py --check
```
