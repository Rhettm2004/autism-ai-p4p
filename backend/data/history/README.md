# Recovered analysis data

These files are the evidence behind numbers already reported in the mid-year
technical report. They were produced by the analysis scripts in `../../../scoring/`,
which lives outside this repository and is not version controlled, so the data
they contain was at risk of being lost or silently regenerated into something
different. They are copied here unmodified, dated by the state they describe, so
every figure and every claim in the final report can be regenerated from tracked
inputs.

Nothing in this directory should be edited. New analysis writes new files
elsewhere; these are a record of what was measured at the time.

| File | What it holds | Used for |
|---|---|---|
| `phase1_results_scored.csv` | 176 Phase 1 responses with the rubric: quality 0-3 and five yes/no flags per response, plus BERTScore and ROUGE-L. Model-assisted annotation, reviewed and agreed by the authors; the `scorer_id` column understates this | Phase 1 safety results, the finding that similarity does not track clinical safety |
| `factual_errors.json` | The 11 responses graded unsafe, each with the specific false statement recorded against its source | The three recurring error classes: wrong age ranges, stale prevalence, reversed scoring |
| `degeneration_metrics.csv` | Per-response repetition measures for the 176 pre-repetition-control responses | The degeneracy baseline that motivated repetition control |
| `retrieval_chunk_ablation.csv` | Recall@1/3/5 and MRR at six chunk sizes, with the resulting passage count | The chunk-size decision, 60 to 120 words |
| `corpus_chunk_probe_ablation.csv` | How many of the seven Phase 1 error probes are covered at each chunk size | The same decision, measured against the errors it needed to fix |
| `corpus_composition_ablation.csv` | Probe coverage and passage count as sources are removed from the corpus | The decision to disable three sources that crowded out answerable content |
| `retrieval_recall_curve.csv` | Recall against k for the final corpus | Retrieval accuracy curve |
| `retrieval_by_category.csv` | Recall@1, Recall@3 and MRR per benchmark category | The finding that retrieval is strongest where the models hallucinated most |
| `retrieval_ranks.csv` | Rank of the correct passage for every benchmark prompt | Per-prompt retrieval diagnosis |
| `router_predictions.csv` | Per-prompt routing predictions from the original single-split evaluation | Historical record, superseded by `../router_eval/predictions_baseline.csv` |
| `router_confusion_matrix.csv` | Confusion matrix from that same single split | Historical record, superseded by `../router_eval/confusion_baseline.csv` |
| `router_single_split_20260721.txt` | Full console output of the original router evaluation, the source of the 65.6% figure | Shows what the single-seed protocol reported, for comparison against the repeated-CV result |

## Superseded numbers

The two router files and the text log describe the single-split protocol used in
July 2026. That protocol is superseded: it reported one accuracy figure with no
interval, and the figure it produced (65.6%) sits below the repeated measurement
of the same unchanged router (68.8% +/- 2.5). They are kept because the final
report describes the progression, and this is what the earlier state actually
looked like.
