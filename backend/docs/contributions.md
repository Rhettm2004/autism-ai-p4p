# Contributions

The final report is an **individual** submission worth 100% of the grade, for a
project carried out jointly, and it requires a signed Statement of Contribution.
This file records who did what while it is still fresh. Reconstructing it in
October from memory would be both painful and inaccurate.

Rayaan Rajabally and Rhett Murdoch, supervised by Dr Seyed Reza Shahamiri,
co-supervised by Rabia Rao. COMPSYS/SOFTENG 700, Project 18.

## How to use this file

Append an entry whenever a piece of work is completed, with the date and, where
it exists, the commit or artefact. Attribution should be specific enough that the
Statement of Contribution can be written from it directly: "designed and built
the retrieval evaluation" rather than "worked on Phase 2".

Entries marked **[confirm]** are inferred from the repository or from the mid-year
report and have not been confirmed by both authors.

## Recorded

| Date | Work | By | Evidence |
|---|---|---|---|
| May 2026 | Phase 1 pipeline: model runner, metrics, config, entry point | Rayaan **[confirm]** | Commit `0775e66` |
| June 2026 | Sourced Q&A benchmark format, reference answers carried through metrics | Rayaan **[confirm]** | Commit `8794bfb` |
| July 2026 | 52-prompt source-matched benchmark, every reference answer quoted from a primary source | **Rhett** | Commit `444d89d` states the set is Rhett's; committed by Rayaan |
| July 2026 | Benchmark version guard, schema validation, archive of earlier versions | Rayaan **[confirm]** | Commit `444d89d` |
| July 2026 | Repetition control and degeneracy detection | Rayaan **[confirm]** | Commits `83d85f2`, `7841485` |
| July 2026 | Phase 3 intent router, first version | Rayaan **[confirm]** | Commit `7841485` |
| July 2026 | RAG pipeline: corpus building, retrieval, wiring into the runner | Rayaan **[confirm]** | Commits `a416253`, `316b0ac`, `d69098e` |
| July 2026 | Corpus curation and retrieval ablations | Rayaan **[confirm]** | Commit `d69098e` |
| July 2026 | Manual rubric grading of all 176 Phase 1 responses | **[confirm — who?]** | `data/history/phase1_results_scored.csv`, column `scorer_id` |
| July 2026 | Mid-year technical report | Both, individually | Two separate submissions |
| August 2026 | Repeated cross-validation harness for the router | Rayaan | Commit `825e971` |
| August 2026 | Re-measurement of the superseded taxonomy | Rayaan | Commit `0d15bc8` |
| August 2026 | Recovery of the analysis data, report figures, exported numbers, notes system | Rayaan | Commits `fca2899` and later |

## To confirm with Rhett

1. Who performed the rubric grading, and whether the second rater will be the
   other author or someone else. This affects both the contribution statement and
   whether the inter-rater check can be claimed as independent.
2. Division of the literature review threads, since both reports need one and
   they must be written independently.
3. Ownership of the Phase 2 generation runs on DeepNet (the 23 July runs).
4. Who is responsible for the Display Day poster and the compendium ReadMe, both
   joint submissions with their own deadlines.

## Note on independence

Both authors submit separate reports on the same project. Shared artefacts (the
repository, the benchmark, the data) are joint; the writing, the framing and the
analysis in each report must be that author's own. Worth agreeing early which
parts of the work each report will foreground, so the two do not read as
paraphrases of each other.
