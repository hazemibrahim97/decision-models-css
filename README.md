# Evaluating Decision Models for Text Annotation in Computational Social Science

Replication package: collection scripts, analysis scripts, and complete per-call records
for the paper's evaluation of decision models against 19 LLM baselines on 18 CSS
classification tasks from Ziems et al. (2024).

## Layout

```
pilot_jev.py               Jev / decisions-endpoint collection (all tasks)
pilot_baselines.py         LLM baseline collection via OpenRouter chat/completions
run_local_decision.py      Qwen3-0.6B RLCD decision model + no-training ablation (local)
run_local_laya.py          Laya 0.4B encoder (local)
run_local_decider.py       decider-0.8b / decider-2b (local)
run_local_semif.py         SemIf-4B frozen logit readout (local, MLX)
jubail_nimble/             Bespoke-Nimble-9B (Slurm, single A100)
run_local_ecosystem.py     Von, Verdict, OpenDecision, Kev-0.8B/4B/9B (one runner, backend argument)
jubail_kev/                Kev and OpenDecision Slurm jobs (single A100)
run_structured_baseline.py Gemini 3.8 Flash structured-output control
parse_llm_answer.py        Free-text answer + verbalized-confidence extraction
results/pilot/             Per-call records, one JSONL per task x model
results/structured/        Per-call records for the structured-output control
analysis/                  All analysis scripts and derived CSVs
figures/                   R/ggplot2 figure scripts
```

## Setup

1. Task data: clone [SALT-NLP/LLMs_for_CSS](https://github.com/SALT-NLP/LLMs_for_CSS)
   into this directory (the scripts read `LLMs_for_CSS/css_data/`).
2. API key: place an OpenRouter key at `~/.config/openrouter/api_key`
   (used for Jev via the decisions endpoint and for all LLM baselines).
3. Python: 3.11+, `numpy`, `scipy`. Local model runners additionally need
   `torch`/`transformers` (and `mlx-lm` for SemIf). Figures need R with
   `ggplot2` and `patchwork`.
4. Open decision systems run through each system's own released code:
   Von from [wfzyx/von](https://github.com/wfzyx/von) (SDK 1.0.1, weights 1.1.0);
   Verdict from [Heman10x-NGU/Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev)
   (commit 30f1556, cloned into this directory; the runner loads `artifacts/v2`);
   OpenDecision from PyPI (`OpenDecision==0.1.1`); Kev from
   [jaredpalmer/kev](https://github.com/jaredpalmer/kev) (commit 557598f, served with
   `kev.serve`; see `jubail_kev/kev_job.sbatch`). The exact version string each system
   reported is stored in every record's `model_resolved` field.

## Reproducing the results

The per-call records in `results/` are the study's raw data; the analysis can be
reproduced from them without any API calls:

```
cd analysis
python build_items.py            # per-item table -> items.csv
python scores.py                 # per-cell metrics -> cell_metrics.csv
python bootstrap.py              # paired bootstrap + BH-FDR -> bootstrap_cells.csv, q1_delta.csv
python routing_curves.py
python reliability_bins.py
python cascade.py
python q1_features.py
python exclusions.py
python appendix_extras.py
python open_models.py
python bootstrap_extras.py       # selection-aware bootstrap + calibration CIs
python structured_baseline.py    # scores the structured-output control
python annotator_disagreement.py <path to wikipedia.annotated.csv>
python make_tables.py            # LaTeX tables -> ../paper/tables/
python make_appendix_tables.py
python make_extras_tables.py
```

`annotator_disagreement.py` needs `wikipedia.annotated.csv` from the
[Stanford politeness corpus](https://www.cs.cornell.edu/~cristian/Politeness.html).

Re-collecting from the APIs (`pilot_jev.py`, `pilot_baselines.py` with
`TASKS=all CONFIDENCE=1`) re-spends real money and, for the LLM baselines,
depends on models and providers that may change or be deprecated; the recorded
resolved model versions and serving providers for every call are in the JSONL
records and in the paper's reproducibility appendix.

## Pre-registration

The design, hypotheses, task grid, and correction procedure were pre-registered
at AsPredicted (#312,511).

## License

MIT (see LICENSE). The task datasets belong to their original creators and are
distributed by Ziems et al.; model outputs in `results/` are provided for
replication of the paper's analyses.
