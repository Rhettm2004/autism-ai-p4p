# Final report requirements

Source: `P4PHandbook2026_ECSE.pdf` (ECSE 2026, governs) and
`final report/FYP_Template-ECSE-2026.docx`.

## Which template governs

The two "Guidelines and Template" PDFs in `final report/` are byte-identical, and
both are the **CEE 2023 civil engineering** version. They do not apply to this
project and their limits conflict with the ECSE handbook.

| | ECSE 2026 handbook (use this) | CEE 2023 PDF (ignore) |
|---|---|---|
| Length | 8,000-13,000 words, typically 12,000; **max 30 pages** | 20 pages |
| Font | Times New Roman 12 pt | Times 11 pt |
| Counted from | Introduction to Future Work inclusive | Introduction to Conclusions |
| Template | `FYP_Template-ECSE-2026.docx` | CEE Word template |

The one thing worth taking from the CEE document is figure practice, which is
sound anywhere: captions below figures, captions above tables, solid fill colours
only, and a 10 pt minimum font size **inside** figures. The figure scripts in this
repository already follow that.

## Deadlines

| Deliverable | Submission | Due |
|---|---|---|
| Display Day poster | Joint | Friday 9 October 2026 |
| **Final research report** | **Individual, 100% of grade** | **Sunday 18 October 2026, 11:59 pm** |
| Research compendium | Joint | Tuesday 20 October 2026 |
| Display Day | Participation | Thursday 22 October 2026 |

Late penalty is 0.5% per hour or part hour, applied to the whole grade. The
handbook advises submitting at least five minutes early.

## Required structure

Sections marked \* count towards both the word count and the page count.

Title Page · Abstract · Signed Statement of Contribution (the Declaration in the
template) · Acknowledgements · Table of Contents · Glossary of Terms ·
Abbreviations · **Introduction\*** · **Literature Review\*** · **middle sections
appropriate to the work\*** · **Discussion\*** · **Conclusions\*** ·
**Future Work\*** · References · Appendices

Notes that change how we write:

- The report is **individual** while the project is joint, so it must be written
  independently and must state contributions. See [`../contributions.md`](../contributions.md).
- The mid-year report explicitly excluded scope, objectives and literature
  review; the final report requires all three.
- No acknowledgement of AI use is required.
- References are the student's choice of style in ECSE; the mid-year report used
  a numbered style, and the template's example is numbered. Stay consistent.

## Research compendium

Joint submission, and effectively this repository. The handbook expects it to
contain supporting material for everything not in the report: experimental
detail, the data behind every plot, and a ReadMe describing its structure,
organisation and contents, such that a future student can reproduce the work.

What that means concretely, and what is already true:

| Expectation | State |
|---|---|
| Data behind every plot | Done. `data/history/`, `data/router_eval/`, `results_thursday/`, exported as tables in `docs/report/numbers/`. |
| Experimental detail | Mostly done. `README.md` covers setup, the run procedure and the corpus; model and generation settings are in `config/`. |
| ReadMe describing structure | Partly. The repository README covers running the pipeline; the compendium needs an orientation document written for someone who has never seen it. |
| Reproducible from scratch | Constrained. The RAG corpus cannot be redistributed and must be rebuilt with `build_corpus.py`. State this explicitly. |
| Analysis scripts | **Gap.** `scoring/` still lives outside the repository and is untracked. The router evaluation, the report figures and the exported numbers have been brought in; the rubric scoring, retrieval evaluation and statistics scripts have not. |

## Page and word budget

30 pages and roughly 12,000 words from Introduction to Future Work, with three
phases to cover. A workable split, to be revisited once drafting starts:

| Section | Words | Pages |
|---|---|---|
| Introduction | 1,000 | 2 |
| Literature Review | 2,500 | 5 |
| Methodology (system, benchmark, metrics, rubric) | 2,000 | 4-5 |
| Phase 1 results | 1,500 | 4 |
| Phase 2 results | 1,800 | 4-5 |
| Phase 3 results | 1,500 | 4 |
| Discussion | 1,200 | 3 |
| Conclusions and Future Work | 800 | 2 |

Figures and tables count towards the page limit but not the word count, which
argues for tables that carry several findings at once rather than one figure per
number.
