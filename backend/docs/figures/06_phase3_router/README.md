# 06_phase3_router

The agentic routing layer: measurement protocol, per-route performance and the progression across changes.

*August 2026, ongoing.*

Rebuild this folder:

```bash
python scripts/make_report_figures.py --group phase3_router
```

**`router_single_split_noise.png`** — Router accuracy across ten cross-validation shuffles of the same 93 prompts. The single split reported at mid-year sits below the distribution it was drawn from.

**`router_per_route_f1.png`** — Router F1 by route with 95% confidence intervals, against the number of labelled examples each route has. Performance tracks training data, not route difficulty.

**`router_confusion.png`** — Router confusion matrix pooled over ten cross-validation shuffles. Rows are the true route, columns the predicted route.

**`router_taxonomy_comparison.png`** — Effect of separating misinformation correction from safety deflection, re-measured under the repeated protocol. The split buys safety recall, not accuracy.

**`router_progression.png`** — Router accuracy at each recorded change, one line per ground-truth track and one panel per dataset, with 95% confidence intervals. The tracks are two scorings of the same router, not two steps.

