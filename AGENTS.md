# AGENTS.md

This is the **runtime fork**: exllamav3 with a HIP decode backend ported to AMD Strix Halo
(`gfx1151`, RDNA 3.5). `main` and `strix-halo` are the same history; `integration` is the
untouched `sdougbrown/exllamav3` base.

- **Setting up a machine to run a model?** Don't start here. Use the recipe, which has the
  scripts and an agent-oriented guide:
  https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-Framework-Strix-Halo-recipe
  (its `AGENTS.md` covers the five ROCm/gfx1151 traps, definition of done, and speed triage).
- **Working on the kernels?** Read `README.strix-halo.md` first — results table, nine
  measured null results (do not re-run them), the LDS bank-conflict analysis, and the
  build/bench-in-separate-shells rule. Harnesses live in `tools/strix_halo/`; `rebuild.sh`
  (`FULL=1` after header edits) is the only supported way to rebuild, because the repo-root
  `exllamav3_ext*.so` shadows site-packages and a hand rebuild will benchmark stale code.
- **Verifying a change**: PPL via `eval/ppl.py -r 20 -l 1024` must stay 4.225935 on the
  Qwen3.8-Flash-Next 3.05 bpw pack; `tools/strix_halo/greedy_ab.py` (run KNOB=x vs KNOB=x
  first for the noise floor) and `tie_check.py` (teacher-forced logit deltas — a summation-
  order change legitimately flips coin-flip tokens; a flip with a multi-logit gap is a bug).
  Then `bench_mtp.py -g` and `prompt_sweep.py` for the six-prompt mean. Never quote a single
  prompt's tok/s as the result.
- Every kernel A/B knob is an env var with the default documented in `README.strix-halo.md`.
  Build-time defines go through `EXL3_HIP_DEFINES`; `EXL3_HIP_STG_PAD` must always be set.
