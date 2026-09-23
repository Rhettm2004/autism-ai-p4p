# 00_overview

Schematics of the system and of how its numbers are produced.

*Drawn 25 August 2026.*

Rebuild this folder:

```bash
python scripts/make_report_figures.py --group overview
```

**`system_architecture.png`** — The pipeline as built. A caregiver turn passes through the Phase 3 router, then retrieval grounds the answer before generation. Every safety constraint sits in the system prompt, which retrieval augments rather than replaces.

**`evaluation_provenance.png`** — How every number in the report is produced. Each box is a tracked file and each arrow a tracked script, so any figure can be traced back to the runs it came from.

