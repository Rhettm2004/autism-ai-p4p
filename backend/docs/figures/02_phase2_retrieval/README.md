# 02_phase2_retrieval

The retrieval layer measured on its own, before any generation. Chunking, corpus composition and per-category recall.

*Measured August 2026.*

Rebuild this folder:

```bash
python scripts/make_report_figures.py --group phase2_retrieval
```

**`retrieval_chunk_ablation.png`** — Retrieval accuracy against chunk size. Recall@1 peaks at 60-word chunks while Recall@3 peaks at 120, which is why the corpus uses 120.

**`corpus_composition_ablation.png`** — Error-probe coverage as crowding sources are removed from the corpus. Cutting the corpus by two thirds improved what it could answer.

**`retrieval_by_category.png`** — Retrieval Recall@1 by benchmark category. Retrieval is strongest on the instrument and boundary questions where the models hallucinated most.

**`phase2_rag_effect.png`** — Automated similarity with and without retrieval, by run (n = 52, 95% CIs). Retrieval lifts every run and pulls the four RAG runs together.

