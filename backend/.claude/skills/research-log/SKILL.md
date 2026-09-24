---
name: research-log
description: Record a change to this research project so it can be used in the final report. Use after making any change that affects results - a router or retrieval change, a new evaluation, a re-measurement, or a recovered finding - to run the measurement, export the numbers, rebuild the figures, append a dated finding to the evidence log, and commit everything together. Also use when asked to "log this", "record this finding", "write this up", or "add this to the notes".
---

# Research log

This project's final report is 100% of the grade and is written in October from
notes taken now. A change whose effect was not measured and recorded cannot be
used in it. This skill is the loop that makes sure every change ends up in the
report's evidence base.

## Before starting

Read `docs/DATA_GUIDE.md` if you have not already this session. It states what
every dataset is and, more importantly, six things that will produce a wrong
sentence in the report if forgotten (benchmark versions, prompt id reuse, what
the automated metrics do not measure, single-rater rubric, ungraded RAG outputs,
the superseded router figure).

## The loop

### 1. Test first

Every change ships with a test in the same commit. For router work:

```bash
python tests/test_router_eval.py
```

A change to how something is measured needs a test of the measurement property
itself, not only of the code path. Reproducibility under fixed seeds, refusal of
invalid comparisons, and guards against silently dropped data are the kinds of
thing worth testing here.

### 2. Measure it

Record the result under a new label, compared against an earlier step on the
same folds:

```bash
python scripts/evaluate_router.py --config <config> --label <label> \
    --compare-to baseline --notes "one line on what changed"
```

Labels are unique and permanent. Never reuse one with `--force` unless the
earlier numbers were never reported anywhere.

Rules that hold regardless of what is being measured:

- No claim from a single split, seed or run. Mean with a 95% confidence interval.
- Comparisons are paired on identical folds, or they are not comparisons.
- If a change makes no measurable difference, that is the finding. Record it.

### 3. Export the numbers

```bash
python scripts/export_report_numbers.py
```

This regenerates `docs/report/numbers/*.csv` and `docs/report/NUMBERS.md` from
tracked data. Numbers used in the report come from here, never from prose.

### 4. Rebuild affected figures

```bash
python scripts/make_report_figures.py --group router
```

Figures are secondary to numbers. Do not build a new figure unless the finding
genuinely needs one; a table usually serves the report better, and tables do not
count against the word limit while figures do count against the page limit.

### 5. Append the finding

Add an entry at the bottom of the relevant log in `docs/evidence/`
(`phase1-baseline.md`, `phase2-rag.md`, `phase3-router.md`). Never edit an
existing finding: if it is wrong, append a correcting one and mark the old one
superseded with a link.

The entry needs:

- The next id in sequence, permanent. `F-P3-007`.
- Date, commit hash, and the artefact the numbers came from.
- The numbers as a table, with confidence intervals.
- What it means, and what it does not show. The second half matters: the
  limitation belongs in the note while it is obvious, not in October when it is
  not.
- Links to related findings and to `decisions.md` where a decision followed.

If the change settles or raises a design question, add or update an entry in
`docs/decisions.md`. If someone other than the committer did the work, add a row
to `docs/contributions.md`.

### 6. Commit everything together

Code, tests, data, notes and figures in one commit, so a note and the numbers it
describes are never separated.

The message is prose explaining why the change was made and what it measured,
with the numbers in it. It is the primary record of what happened.

**No co-authorship trailers, and no mention of Claude, Anthropic or any AI tool
anywhere in the commit, the code or the notes.**

Do not commit to `main` unless asked, and never push unless asked.

### 7. Sync when convenient

```bash
python scripts/sync_report_notes.py
```

Exports a snapshot of the notes, numbers and figures to `final report/notes/` for
writing. The destination is disposable and must never be edited.

## What good looks like

A finding someone else could act on without asking a question: what was believed
before, what changed, what it measured, with an interval, and what it still
cannot show.
