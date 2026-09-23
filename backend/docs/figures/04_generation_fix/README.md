# 04_generation_fix

The truncation fault and its repair, measured mechanically rather than by rubric.

*Diagnosed and fixed 24-25 August 2026.*

Rebuild this folder:

```bash
python scripts/make_report_figures.py --group generation_fix
```

**`p2_generation_fix.png`** — Effect of completion-scoped repetition control on the generated output. Mistral's retrieval responses stop being cut off and recover the length the constraint was costing them; nothing else moves.

**`p2_length_recovery.png`** — Length of Mistral's retrieval responses before and after the fix. The spike of answers under 50 tokens, which is what a banned continuation produced, is gone.

