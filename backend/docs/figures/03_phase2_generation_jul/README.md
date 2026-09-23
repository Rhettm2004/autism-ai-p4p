# 03_phase2_generation_jul

The first graded comparison of baseline against retrieval, on the 23 July runs. Superseded as the headline by folder 05, kept because the report describes the progression.

*Runs 23 July 2026, graded 24 August 2026.*

Rebuild this folder:

```bash
python scripts/make_report_figures.py --group phase2_generation_jul
```

**`p2_hallucination_by_run.png`** — Hallucination rate before and after retrieval, by run (n = 52 per bar). Every run falls, and every fall is significant on a paired test.

**`p2_hallucination_by_category.png`** — Hallucination rate by prompt category, before and after retrieval. Retrieval clears the instrument and recognition questions and barely touches general health knowledge.

**`p2_truncation_confound.png`** — The generation-budget fault. Retrieval truncates a third of Mistral's responses and none of Llama's, and that alone explains Mistral's apparent quality loss under retrieval.

**`p2_rubric_overview.png`** — Every rubric measure, baseline against retrieval, pooled over all 416 responses. Accuracy improves, diagnostic safety is unchanged at zero, and two measures move the wrong way.

