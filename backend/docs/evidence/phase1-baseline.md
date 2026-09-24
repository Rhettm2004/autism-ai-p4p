# Phase 1 evidence log: baseline LLM behaviour

Append-only. Newest finding at the bottom. Ids are permanent.

Phase 1 answers RQ1: can a plain LLM, with no extra sources, help a caregiver
through autism screening? Mistral-7B and Llama3-8B, both 4-bit quantised, at
temperature 0.1, each run zero-shot and few-shot over the 44-prompt benchmark:
176 responses, every one read and graded by hand.

Most of this log is backfilled from the mid-year report and the recovered
analysis data in `data/history/`. Findings recorded after August 2026 were
measured directly.

---

## F-P1-001: Instruction-level safety held completely

**Artefact.** `data/history/phase1_results_scored.csv`, column
`diagnostic_overreach_flag`.

Not one of the 176 responses diagnosed a child, under any condition, including
every adversarial strategy in the benchmark: asking outright, instructing the
model to ignore its instructions, claiming to be a paediatrician, asking
hypothetically, and asserting the child was already known to be autistic.
Diagnostic overreach was 0% in all four runs.

The two adversarial categories also scored highest of any group on quality (2.92
and 2.58 out of 3) with no factual errors.

This is the single strongest Phase 1 result and it is the foundation of the
Phase 3 design argument: written instructions held a hard boundary where a
learned component would give no such guarantee.

---

## F-P1-002: Factual accuracy is the weak point

**Artefact.** `data/history/phase1_results_scored.csv`;
`data/history/factual_errors.json`.

| Run | Mean quality (0-3) | Hallucination rate | 95% CI | Addresses question |
|---|---|---|---|---|
| Mistral-7B zero-shot | 2.30 | 18.2% | [9.5%, 32.0%] | 95.5% |
| Mistral-7B few-shot | 2.36 | 18.2% | [9.5%, 32.0%] | 97.7% |
| Llama3-8B zero-shot | 2.43 | 9.1% | [3.6%, 21.2%] | 88.6% |
| Llama3-8B few-shot | 2.50 | 15.9% | [7.9%, 29.4%] | 90.9% |

Pooled, 15.3% of responses contained something factually wrong, above the 10%
review threshold set in advance. Eleven of 176 scored 0 because the error could
mislead a caregiver. Mean quality 2.40 of 3.

No run is measurably better than another: the confidence intervals overlap
heavily, hallucination counts between models give p = 0.40, quality p = 0.31.
Neither model is safer than the other on this data.

---

## F-P1-003: The errors are concentrated, and they are about the instruments

**Artefact.** `data/history/phase1_results_scored.csv`, grouped by category;
`data/history/factual_errors.json` for the verbatim errors.

Questions about the screening tools themselves were wrong 43.8% of the time and
scored worst on quality (1.66 of 3). The two safety categories had no factual
errors at all.

Three error classes recur across both models:

1. **Wrong age ranges.** Q-CHAT-10 placed at 4-9 years and at 3-6 months, against
   roughly 18-24 months. M-CHAT placed at 2-12 years and at 18-24 months, against
   16-30 months.
2. **Stale prevalence.** 1 in 44 quoted, against the current CDC figure of 1 in 31.
3. **Reversed scoring.** A Q-CHAT-10 score of 6 described as typical or below
   threshold when the cutoff is 3, which tells a high-risk caregiver that nothing
   is wrong.

The third is the most dangerous class of error in the whole project: it is
confident, specific, and it points a worried parent away from help. It is also
exactly the kind of error a retrieved source fixes, which is the argument that
carries Phase 1 into Phase 2.

---

## F-P1-004: The two models fail in opposite directions

**Artefact.** `data/history/phase1_results_scored.csv`, columns
`addresses_question_flag` and `hallucination_flag`.

Mistral answers confidently but wrongly, addressing the question 95.5-97.7% of the
time with the higher hallucination rate. Llama fails by over-refusing, treating
ordinary screening questions as diagnosis requests and answering only 88.6-90.9%
of the time. One example: Llama declined to correct a claim about high-dose
vitamins.

Both are wrong for a caregiver-facing tool, and they need opposite fixes: give
the first sources to quote, and teach the second when it is safe to answer. This
finding is the reason the Phase 3 taxonomy separates misinformation correction
from safety deflection — see [F-P3-005](phase3-router.md).

---

## F-P1-005: Automated similarity does not track clinical safety

**Artefact.** `data/history/phase1_results_scored.csv`, columns `bertscore_f1`
and `quality_score`.

Because the same 176 responses carry both automated scores and manual grades, the
two can be compared directly.

| Metric | r with quality | p | Mean, all 176 | Mean, the 11 unsafe | Unsafe above the overall mean |
|---|---|---|---|---|---|
| BERTScore F1 | 0.084 | 0.267 | 0.8655 | 0.8627 | 4 of 11 |
| ROUGE-L | **−0.030** | 0.690 | 0.2014 | **0.2294** | **6 of 11** |

Recomputed 24 August 2026 from the recovered rubric file; see
`docs/report/numbers/p1_similarity_vs_quality.csv`. The BERTScore figures
reproduce the mid-year report exactly.

The ROUGE-L result is worse than the report states and worth saying explicitly:
the correlation is very slightly *negative*, and the eleven responses graded
unsafe score **higher** on ROUGE-L than the overall mean (0.2294 against 0.2014),
with six of the eleven above it. Optimising for ROUGE-L on this benchmark would,
if anything, select against safety.

This is what BERTScore does: a response with the wrong age range still means
roughly the same thing as one with the right age range, because every word around
the number is identical. The consequence is methodological and it propagates
through the whole project. Rubric quality is the primary measure; BERTScore and
ROUGE-L are descriptive only, and no model selection decision rests on them.

---

## F-P1-006: Degeneracy was present and invisible to the metrics

**Artefact.** `data/history/degeneration_metrics.csv`. **Commit.** `83d85f2`.

Phase 1 ran with no repetition control.

| Measure | Value |
|---|---|
| Responses that collapsed into a loop (strict threshold) | 2 of 176 |
| Responses containing at least one repeated 5-gram | 52 of 176 (29.5%) |
| Llama3-8B repeated-token fraction, zero-shot | 3.2× Mistral-7B |
| BERTScore of one degenerate response | 0.862 against an overall mean of 0.866 |

The degenerate response scored at the mean because its opening sentence was well
formed and the embedding is dominated by shared topic words. A second independent
reason not to trust similarity scores as a safety measure.

Fixed by setting `repetition_penalty` 1.15 and `no_repeat_ngram_size` 6,
identically for both models so the setting cannot become a confound, matching the
existing treatment of temperature. Both are passed only when present in config,
so removing them reproduces original Phase 1 behaviour exactly. After the change:
**0 degenerate responses across all 416 Phase 2 outputs**.

---

## F-P1-007: The benchmark changed three times, and one version was nearly lost

**Commits.** `0775e66`, `8794bfb`, `444d89d`.

| Version | Prompts | Status |
|---|---|---|
| v1 | 40 | `data/benchmark/archive/phase1_baseline_benchmark_v1_40prompt.csv` |
| v2 | 44 | Reconstructed from the `_with_metrics` result files; the original was never committed |
| v3 | 52 | Current. Every reference answer quoted verbatim from a primary source (CDC, M-CHAT FAQ, NIMH, WHO, NICHD) |

Two things here matter for the report's methodology section.

First, the human-graded results (176 responses) come from v2, while all automated
results come from v3. The two cannot be pooled, and any statement that mixes them
is wrong.

Second, benchmark revisions reuse prompt ids (P001, P002, ...) for different
questions. An id-only join between results and benchmark therefore succeeds while
scoring every response against an unrelated reference answer, producing plausible
and meaningless metrics. `run_auto_metrics` now compares prompt text and refuses
to proceed on a mismatch. This is a good example for the report of an evaluation
bug that produces no error and no obviously wrong output.

---

## F-P1-008: How the rubric grades were actually produced

**Status.** Open, but narrower than previously recorded.

**Corrected 24 August 2026**, twice, and the second correction matters.

The grades were produced by an **LLM-assisted annotation pass, then reviewed and
agreed by the authors**. They are not raw model output, and they are not
independent human scoring either.

The data understates this. Every row of `phase1_results_scored.csv` carries
`scorer_id = "LLM-assisted (Claude), pending human validation"`, and
`scoring/annotations.py` describes the annotations as preliminary and awaiting
validation. That review did happen; the metadata was never updated to say so, and
`scoring/score_responses.py` still writes the stale string on every run. Anyone
reading the data without asking would conclude, as this log briefly did, that no
human had looked at it.

**What is established.** Each of the 176 annotations was reviewed by a human and
agreed. Each hallucination flag is tied to a stated, checkable factual claim
recorded in a note, which is what makes review quick and is why
`p1_factual_errors.csv` can list the exact false statements.

**What is not established.** The rubric in `files/phase1_scoring_rubric.md`
specifies two scorers grading *independently*, calibrating on five shared
prompts, and resolving any disagreement greater than one point. That is not what
was done, so **there is no inter-rater agreement figure**, and there cannot be
one computed after the fact from an agreement-based review.

**Why the distinction matters, in one sentence for the report.** Reviewing a
proposed label is anchored on that label: a reviewer accepts more than they would
have produced unprompted, so agreement under review is an upper bound on what
independent scoring would have found.

**The cheap fix.** One author scores a stratified sample of 30 to 40 responses
independently, without seeing the existing annotations, and agreement is computed
against them. That produces the statistic the rubric asks for, at an hour of
work, and it converts the limitation into a measured quantity. Tracked as
[D-105](../decisions.md).

**Report wording.** The mid-year report says every answer was read and graded
with the reference answer and its source open beside it. That is true of the
review pass and would be read as independent human grading. The final report
should describe the procedure as it was: model-assisted annotation with human
adjudication, no independent second scorer.
