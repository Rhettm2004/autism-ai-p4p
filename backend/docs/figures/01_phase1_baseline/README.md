# 01_phase1_baseline

The ungrounded baseline: what the models did with no retrieval and no repetition control.

*Runs July 2026, graded by hand.*

Rebuild this folder:

```bash
python scripts/make_report_figures.py --group phase1_baseline
```

**`phase1_quality_distribution.png`** — Distribution of manual rubric quality scores by run (n = 44 per run). Scores are 0 unsafe to 3 strong.

**`phase1_hallucination_by_category.png`** — Hallucination rate by prompt category, all four Phase 1 runs pooled (n = 176). Errors concentrate on facts about the screening instruments.

**`phase1_bertscore_vs_quality.png`** — BERTScore F1 against manual rubric quality for all 176 Phase 1 responses. The flat trend shows semantic similarity does not track clinical safety.

**`phase1_degeneracy.png`** — Repetition in the 176 Phase 1 responses generated without repetition control, by run. Llama3-8B repeats far more than Mistral-7B in the zero-shot condition.

