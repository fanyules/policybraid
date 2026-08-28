# PolicyBraid

PolicyBraid asks whether heterogeneous rollout runtimes implement stable,
prompt-dependent behavior policies that measurably change the canonical GRPO
learner update at a fixed checkpoint.

The current and only scientific gate is **PM-A** on A100 and patched Ascend
910B. It freezes Qwen3-1.7B, matched prompt groups, execution context, sampling,
and LoRA gradient coordinates before comparing the two runtimes. Controller
design, online LoRA transfer, first-K networking, consistency mode, and RTX are
out of scope until PM-A passes.

- Protocol: `docs/PM_POLICY_CENSUS_PROTOCOL.md`
- Frozen configuration: `configs/pm_a.json`
- Runtime prerequisite note: `docs/RUNTIME_PREREQUISITE.md`
- Candidate workload generator: `scripts/build_pm_a_candidates.py`
- Screening adjudicator: `scripts/screen_pm_a_prompts.py`

The former vLLM-Ascend investigation is intentionally absent from `main`.
Its full evidence remains recoverable on the
`codex/policymesh-runtime-reentry` branch; the older unpatched Gate record is on
`archive/policymesh-g0-failure`. Only the qualified runtime interface and its
provenance are prerequisites here.

## Local checks

```bash
python -m unittest discover -s tests -v
python scripts/build_pm_a_candidates.py --check
```

