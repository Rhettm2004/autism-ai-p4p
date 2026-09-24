# 05_phase2_rerun_aug

The re-run after the three fixes, graded blind. This is the pass the report should quote for the retrieval effect.

*Runs and grading 25 August 2026.*

Rebuild this folder:

```bash
python scripts/make_report_figures.py --group phase2_rerun_aug
```

**`rerun_rubric_comparison.png`** — Every rubric measure before and after the three fixes, baseline against retrieval. The two measures that moved the wrong way in Phase 2 no longer do, and retrieval-arm quality gains a quarter of a point.

**`rerun_error_families.png`** — Where the recurring errors came from. Every response that misused a screening instrument was a baseline response; retrieval eliminated that family entirely. Fabricated citations and refusals split evenly and are not a retrieval effect.

**`rerun_hallucination_by_category.png`** — Hallucination by prompt category after the fixes, baseline against retrieval. Retrieval clears the instrument and screening categories and still cannot touch general health knowledge.

**`rerun_quality_distribution.png`** — Where the quality gain comes from. Retrieval roughly doubles the share of responses scoring 3 and halves the share scoring 1, rather than shifting every response a little.

**`rerun_hallucination_by_run.png`** — Hallucination rate before and after retrieval, by run, on the 25 August data (n = 52 per bar). Every run falls and every fall is significant on a paired test, so the effect is not one model or one prompting condition.

**`followup_targeting_collapse.png`** — Follow-up advice by whether the turn owed a next step. Both arms post the same flat rate; only the baseline advises the turns that owe one more often than the turns that do not.

