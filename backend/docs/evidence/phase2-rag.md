# Phase 2 evidence log: retrieval-augmented generation

Append-only. Newest finding at the bottom. Ids are permanent.

Phase 2 answers RQ2: if the model is given clinical sources to quote from, does it
stop getting facts wrong? Retrieval can be evaluated before the model runs, and
nothing downstream works if retrieval is broken, so it was tested first.

The pipeline splits each source document into passages, indexes them with TF-IDF,
and injects the closest passages into the prompt ahead of the safety
instructions. The test uses the source label already recorded against every
benchmark prompt: give the retriever the caregiver's question and see whether the
passage carrying the correct answer comes back near the top.

---

## F-P2-001: Retrieval baseline on the 52-prompt benchmark

**Artefact.** `data/history/retrieval_recall_curve.csv`,
`data/history/retrieval_ranks.csv`.

| Measure | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| TF-IDF | 50.0% | 57.7% | 71.2% | 76.9% | 0.584 |

Half the questions get the right passage first, rising to 71.2% within the top
five. Phase 2 retrieves five passages rather than three as a direct consequence
of the gap between those two numbers.

TF-IDF ranks a passage by how many of the question's rarer words it contains,
with no sense of meaning. That is a deliberate starting point, not a limitation
we failed to notice: it is deterministic, inspectable, needs no GPU, and it
gives a floor that a dense retriever has to beat to justify itself.

---

## F-P2-002: Retrieval is strongest exactly where the models hallucinated

**Artefact.** `data/history/retrieval_by_category.csv`. Compare against
[F-P1-003](phase1-baseline.md).

| Category | n | Recall@1 | Recall@3 | MRR |
|---|---|---|---|---|
| diagnosis_boundary | 5 | 100% | 100% | 1.000 |
| screening_item_clarification | 3 | 100% | 100% | 1.000 |
| screening_limitations | 2 | 100% | 100% | 1.000 |
| next_steps_referral | 3 | 66.7% | 66.7% | 0.678 |
| caregiver_support | 2 | 50.0% | 100% | 0.750 |
| screening_tool_use | 2 | 50.0% | 100% | 0.750 |
| general_health_knowledge | 5 | 40.0% | 80.0% | 0.606 |
| treatment_support | 5 | 40.0% | 40.0% | 0.520 |
| screening_follow_up | 6 | 33.3% | 33.3% | 0.382 |
| safety_vaccines | 3 | 33.3% | 33.3% | 0.426 |
| general_autism_knowledge | 13 | 30.8% | 30.8% | 0.380 |

The three categories retrieval answers perfectly are instrument and boundary
questions, which use distinctive vocabulary. The two worst are general knowledge
and follow-up, which share vocabulary with everything else in the corpus.

This is the sharpest argument in Phase 2 and it should be a figure in the report:
retrieval is most accurate precisely on the questions Phase 1 got most wrong.
The categories with 43.8% hallucination in Phase 1 are the ones with 100%
Recall@1 here.

---

## F-P2-003: Chunk size decides whether a fact is reachable

**Artefact.** `data/history/retrieval_chunk_ablation.csv`,
`data/history/corpus_chunk_probe_ablation.csv`. **Commit.** `316b0ac`.

| Chunk (words) | Overlap | Passages | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| 30 | 8 | 245 | 44.2% | 61.5% | 71.2% | 0.555 |
| 40 | 10 | 188 | 46.2% | 61.5% | 75.0% | 0.573 |
| 60 | 15 | 123 | **50.0%** | 57.7% | 71.2% | 0.584 |
| 80 | 20 | 97 | 48.1% | 63.5% | 71.2% | 0.589 |
| 120 | 30 | 72 | 48.1% | **67.3%** | 71.2% | 0.587 |
| 200 | 50 | 53 | 48.1% | 65.4% | 69.2% | 0.591 |

Recall@1 peaks at 60 words and Recall@3 at 120, so the aggregate scores do not
settle the choice. What settled it was checking the ablation against the seven
specific Phase 1 errors the corpus exists to fix: at 60 words the CDC prevalence
table split across a chunk boundary and the passage carrying the current 1 in 31
figure fell to rank 5, outside a top-3 window, despite the fact being indexed.
Probe coverage rises from 3 of 7 at 60 words to 4 of 7 at 120.

The default is 120 words. The general lesson for the report: an aggregate
retrieval metric can be flat across a parameter that decides whether a specific
required fact is reachable at all.

---

## F-P2-004: A smaller corpus retrieved better

**Artefact.** `data/history/corpus_composition_ablation.csv`. **Commit.** `d69098e`.

| Corpus | Passages | Probes covered at k=3 | at k=5 |
|---|---|---|---|
| All 20 sources | 425 | 4 of 7 | 4 of 7 |
| Drop raw instrument forms | 405 | 4 of 7 | **6 of 7** |
| Drop forms + academic paper | 283 | **5 of 7** | 6 of 7 |
| Drop forms + paper + NZ summary | 165 | 5 of 7 | 6 of 7 |

Three sources measurably hurt retrieval and were disabled: the Allison 2012
paper (122 passages, 29% of the corpus, mostly methods and references), and the
raw Q-CHAT questionnaire and scoring grid, which matched queries lexically
without carrying answerable content. A questionnaire form contains the exact
words of the question a caregiver asks and none of the answer, which is the worst
possible profile for a lexical retriever.

Cutting the corpus by a third improved what it could answer. That is a
counter-intuitive, quotable result: for TF-IDF retrieval, corpus quality
dominates corpus size.

---

## F-P2-005: One document can occupy every retrieved slot

**Artefact.** commit `d69098e`; implemented as `max_per_source` in
`src/retrieval.py`.

On the 425-passage corpus the M-CHAT FAQ took all six top slots for a scoring
question, while the authoritative scoring page ranked 20th. A long document with
consistent vocabulary crowds out shorter, more authoritative pages.

`Retriever.retrieve` now caps passages per source at 2 by default. Without the
cap, "retrieve the top 5" can mean "retrieve five passages of one document",
which reduces the effective evidence the model sees to a single source.

---

## F-P2-006: Sourcing problems that shaped the corpus

**Commit.** `d69098e`.

Three practical findings worth a paragraph in the methodology:

1. **cdc.gov returns 403 to scripted requests** regardless of user agent, so the
   prevalence page is saved by hand under `data/corpus/manual/`.
2. **Some required facts are never stated directly.** The M-CHAT 16-30 month
   range is absent from all 23 fetched M-CHAT passages, and the Q-CHAT 18-24
   month range appears only indirectly. Short sourced notes were added under
   `manual/` so those facts are reachable.
3. **Licensing limits redistribution.** M-CHAT and NICE permit research use but
   not redistribution, so the built corpus is gitignored. Only the manifest of
   sources and the compiled reference notes are committed, which is a constraint
   on how reproducible the corpus can be made for the compendium.

---

## F-P2-007: RAG generation results across eight runs

**Artefact.** `results_thursday/summary_table.csv`, `results_thursday/SUMMARY.txt`;
raw responses in the same directory.

Both models, both conditions, with and without retrieval: 8 runs, 416 responses,
all on the 52-prompt benchmark.

| Run | BERTScore F1 | ±95% CI | ROUGE-L F1 | ±95% CI |
|---|---|---|---|---|
| Mistral-7B zero-shot, baseline | 0.861 | 0.005 | 0.171 | 0.012 |
| Mistral-7B zero-shot, RAG | 0.866 | 0.010 | 0.195 | 0.038 |
| Mistral-7B few-shot, baseline | 0.860 | 0.005 | 0.170 | 0.014 |
| Mistral-7B few-shot, RAG | 0.867 | 0.011 | 0.207 | 0.036 |
| Llama3-8B zero-shot, baseline | 0.839 | 0.004 | 0.132 | 0.010 |
| Llama3-8B zero-shot, RAG | 0.867 | 0.007 | 0.224 | 0.028 |
| Llama3-8B few-shot, baseline | 0.844 | 0.004 | 0.140 | 0.011 |
| Llama3-8B few-shot, RAG | 0.868 | 0.007 | 0.225 | 0.033 |

Paired, same prompts:

| Run | Metric | Δ | t | dz | p |
|---|---|---|---|---|---|
| Mistral-7B zero-shot | BERTScore | +0.005 | +1.07 | 0.15 | 0.289 n.s. |
| Mistral-7B zero-shot | ROUGE-L | +0.024 | +1.26 | 0.17 | 0.214 n.s. |
| Mistral-7B few-shot | BERTScore | +0.006 | +1.36 | 0.19 | 0.181 n.s. |
| Mistral-7B few-shot | ROUGE-L | +0.038 | +2.17 | 0.30 | 0.035 * |
| Llama3-8B zero-shot | BERTScore | +0.028 | +8.27 | 1.15 | 5.5e-11 * |
| Llama3-8B zero-shot | ROUGE-L | +0.093 | +6.48 | 0.90 | 3.6e-08 * |
| Llama3-8B few-shot | BERTScore | +0.024 | +6.10 | 0.85 | 1.4e-07 * |
| Llama3-8B few-shot | ROUGE-L | +0.085 | +5.18 | 0.72 | 3.8e-06 * |

Retrieval fired on 208 of 208 RAG responses at a mean latency of 1.7 ms.

Two patterns matter. The gain is large for Llama3-8B and negligible for
Mistral-7B, which follows from where each started: Llama began lowest and was the
model that over-refused and missed instrument facts, so quoting a source closed
most of that gap, while Mistral had little headroom. And the four RAG runs
converge into a narrow band (BERTScore 0.866-0.868, ROUGE-L 0.195-0.225): the
between-model gap that looked decisive in Phase 1 essentially disappears once
both models can quote sources.

---

## F-P2-008: What the Phase 2 numbers cannot decide

**Status.** Open, and it blocks the model-selection decision.

1. These are automated similarity scores, and [F-P1-005](phase1-baseline.md)
   showed they do not track clinical safety. **No rubric grading of the RAG
   outputs has been done.** Until it is, Phase 2 cannot claim that retrieval made
   the system safer, only that it moved the output towards the reference text.
2. ROUGE-L rises partly by construction: the v3 reference answers are quoted
   verbatim from primary sources, so a model quoting the same source scores
   higher regardless of whether its answer is better for a caregiver.
3. The hallucination rate, the measure that actually motivated Phase 2, has not
   been recomputed on the RAG outputs at all.

The honest one-line summary of Phase 2 so far: retrieval reliably injects the
right sources at negligible cost and moves output in the intended direction, and
whether that translates into fewer dangerous errors is still unmeasured.

---

## F-P2-009: Retrieval cuts hallucination by two thirds

**Date.** 24 August 2026. **Artefacts.** `data/grading/phase2_results_scored.csv`,
`data/grading/phase2_rubric_paired.csv`, per-response annotations in
`data/grading/annotations/`. Method and blinding in `src/rubric.py`.

All 416 Phase 2 responses have been graded against the Phase 1 rubric, blind to
model, condition and retrieval status, in a shuffled order, with the mapping back
to the runs opened only after the last batch was finished. This answers RQ2,
which [F-P2-008](#f-p2-008-what-the-phase-2-numbers-cannot-decide) recorded as
unanswerable.

### Headline

| Measure | Baseline | RAG | Δ | p | dz |
|---|---|---|---|---|---|
| **Hallucination** | **38.0%** | **11.1%** | **−26.9pp** | 6.9e-12 | −0.50 |
| Quality (0-3) | 1.98 | 2.14 | +0.16 | 0.032 | 0.15 |
| **Diagnostic overreach** | **0.0%** | **0.0%** | 0 | — | — |
| Addresses the question | 95.7% | 85.1% | −10.6pp | — | — |
| Recommends professional follow-up | 27.9% | 18.8% | −9.1pp | — | — |
| Truncated mid-sentence | 6.2% | 17.3% | **+11.1pp** | 0.00019 | 0.26 |

Paired by prompt, 208 pairs. **The hallucination reduction holds in all four runs
separately**, at p = 0.0003, 1.3e-05, 0.005 and 0.006.

**This is the Phase 2 result the project was built to produce.** Retrieval does
not merely move output closer to the reference text, which is all
[F-P2-007](#f-p2-007-rag-generation-results-across-eight-runs) could establish.
It removes two thirds of the factual errors a caregiver would be exposed to.

### Where the errors went

Hallucination rate by category, baseline against RAG, pooled over both models and
conditions:

| Category | n | Baseline | RAG |
|---|---|---|---|
| recognition_and_diagnosis | 8 | **100%** | **0%** |
| screening_limitations | 16 | **75%** | **0%** |
| screening_tool_use | 16 | 75% | 25% |
| general_autism_knowledge | 104 | 54% | 15% |
| diagnosis_boundary | 40 | 35% | 5% |
| screening_follow_up | 48 | 33% | 4% |
| next_steps_referral | 24 | 33% | 0% |
| safety_vaccines | 24 | 33% | 17% |
| general_health_knowledge | 40 | 35% | 30% |
| screening_item_clarification | 24 | 8% | 8% |
| **safety_treatment** | 8 | **0%** | **50%** |

**Retrieval works exactly where Phase 1 said it would.** The categories it clears
completely are the instrument and recognition questions: the wrong age ranges,
the reversed scoring, the invented rescreening schedules. That was the argument
for building a corpus, made in [F-P1-003](phase1-baseline.md) and
[F-P2-002](#f-p2-002-retrieval-is-strongest-exactly-where-the-models-hallucinated),
and it is now demonstrated on generated output rather than on retrieval ranks.

**General health knowledge barely moves, 35% to 30%.** These are the
mitochondrial-disease questions, and they are the category where retrieval scored
worst in F-P2-002. Retrieval quality and generation quality track each other.

**Medication safety gets worse, 0% to 50%.** Only 8 responses, so this is four
responses against four, but it is the wrong direction on a safety-adjacent
category and both errors are about FDA approval: one response claims approval
covers irritability, hyperactivity and sleep, another claims no approval exists
at all. Worth watching rather than concluding from.

### The confound, found and controlled

Truncation is not evenly distributed. It is a **Mistral-with-retrieval** problem:

| Run | Baseline truncated | RAG truncated |
|---|---|---|
| llama3-8b few-shot | 3.8% | **0.0%** |
| llama3-8b zero-shot | 3.8% | **0.0%** |
| mistral-7b few-shot | 7.7% | **36.5%** |
| mistral-7b zero-shot | 9.6% | **32.7%** |

Retrieval injects five passages into the prompt. On Mistral that consumes enough
of the context or generation budget that a third of responses stop mid-sentence;
on Llama it does not. This is a configuration fault in our pipeline, not a
property of retrieval, and it is the explanation for the one result that looked
like retrieval hurting:

| Run | Quality, all responses | Quality, non-truncated only |
|---|---|---|
| llama3-8b few-shot | 2.02 → **2.44** | 2.02 → 2.44 |
| llama3-8b zero-shot | 1.94 → **2.50** | 1.94 → 2.50 |
| mistral-7b few-shot | 2.00 → **1.85** | 2.06 → **2.15** |
| mistral-7b zero-shot | 1.96 → **1.77** | 2.00 → **2.00** |
| Pooled | 1.98 → 2.14 | 2.01 → **2.31** |

**Mistral's apparent quality loss under retrieval disappears entirely once
truncated responses are excluded.** The same explanation covers the drop in
answering the question, 95.7% to 85.1%, which is concentrated in the two Mistral
RAG runs at 78.8% and 73.1%.

The honest statement for the report is that retrieval improved quality in every
run, and that our generation settings then gave a third of that improvement back
on one model by cutting its answers off. Re-running Mistral with a larger
generation budget is the obvious next step, and it should be done before the
report quotes the quality figures.

### Two costs that are not artefacts

**Professional follow-up advice falls, 27.9% to 18.8%.** RAG responses quote the
source and stop. The rubric counts a recommendation to see a professional as a
presence check, and retrieval makes it less likely. For a caregiver-facing tool
that matters: the system becomes more accurate and slightly less useful at
telling someone what to do next.

**The screening-versus-diagnosis distinction also falls slightly**, 8.2% to 6.2%,
though both figures are low enough that the detector may simply be missing the
phrasing.

### What still holds

**Diagnostic overreach is zero across all 416 responses**, with and without
retrieval, replicating the Phase 1 finding on a second benchmark, a second set of
prompts and a larger sample. That is now 592 graded responses across both phases
with no response crossing the diagnosis line.

### Limits

Grades are model-assisted with human adjudication and there is no independent
second scorer, so no inter-rater agreement figure exists
([F-P1-008](phase1-baseline.md), [D-105](../decisions.md)). Blinding to condition
was maintained throughout, which the Phase 1 pass did not do, but blinding is
imperfect: a response quoting the M-CHAT FAQ verbatim is recognisably a retrieval
response. That leakage would inflate the effect if it operated, and the effect is
large enough that it is unlikely to be the whole story, but it cannot be ruled
out. An independent human pass over a stratified sample would settle both points.

Resolves [D-104](../decisions.md).

### Addendum, 24 August 2026: the four figures for this finding

Registered in `scripts/make_report_figures.py` under group `phase2_rubric` and
built into `docs/figures/`. All four read `data/grading/phase2_results_scored.csv`,
so none of them can drift from the table above.

| Figure | What it shows | Use it for |
|---|---|---|
| `p2_hallucination_by_run.png` | Hallucination rate baseline against RAG, one bar pair per run plus the pooled pair, with the drop in percentage points above each pair | The RQ2 headline. The point is that all four runs fall, so the effect is not one model or one prompting condition |
| `p2_hallucination_by_category.png` | Dumbbell per category, baseline dot to RAG dot, sorted by baseline rate, with n on the right | Where the errors went. Green lines are reductions, the one red line is `safety_treatment` going the wrong way on n = 8 |
| `p2_truncation_confound.png` | Left: truncation rate per run, baseline against RAG. Right: quality per run for baseline, RAG, and RAG excluding truncated responses | The caveat that must travel with the quality figures. The right panel is the argument: excluding truncated responses, RAG is above baseline in every run |
| `p2_rubric_overview.png` | Every rubric measure pooled, baseline against RAG, with the signed delta coloured by whether it is an improvement | The honest one-figure summary. Two measures move the wrong way and the figure says so |

**Reading `p2_hallucination_by_category.png` next to `retrieval_by_category.png`
is the strongest argument in Phase 2.** Retrieval quality and generation accuracy
rank the categories the same way: the categories where retrieval ranked the right
passage highly are the categories where hallucination went to zero, and
`general_health_knowledge`, the worst-retrieved category, is the one that barely
moved. That is the causal story rather than a correlation between two summaries.

**Do not use `p2_rubric_overview.png` without `p2_truncation_confound.png`.**
Two of the three measures that move the wrong way in the overview, answering the
question and cutting off mid-sentence, are the same fault seen twice, and it is
ours, not retrieval's.

---

## F-P2-010: The truncation was our repetition control, not the token budget

**Date.** 24 August 2026. **Artefacts.** `data/history/truncation_diagnosis.csv`,
`data/history/truncation_diagnosis_completion_scope.csv`. Produced by
`python scripts/diagnose_truncation.py` and `--scope completion`.

**This supersedes the explanation given in
[F-P2-009](#f-p2-009-retrieval-cuts-hallucination-by-two-thirds), which attributed
the truncation to the generation budget, and it resolves
[D-109](../decisions.md), which proposed re-running Mistral with a larger budget.
A larger budget would have changed nothing.**

### The first explanation was wrong

No Mistral RAG response reached `max_new_tokens`. The cap is 512. The truncated
responses average **42 generated tokens**, and the longest Mistral RAG response
in either condition is 389. The models were stopping early on their own, so the
budget was never the binding constraint.

| Run | max completion tokens | mean | responses at the 512 cap |
|---|---|---|---|
| mistral-7b few-shot RAG | 389 | 64 | 0 of 52 |
| mistral-7b zero-shot RAG | 230 | 65 | 0 of 52 |
| llama3-8b few-shot RAG | 422 | 183 | 0 of 52 |
| llama3-8b zero-shot RAG | 404 | 204 | 0 of 52 |

Two baseline runs *did* hit the cap, 3 and 4 responses of 52. Truncation from the
budget exists in this data, but it is a baseline phenomenon and it is small.

### What was actually happening

Generation runs with `no_repeat_ngram_size: 6`. In HuggingFace that constraint
reads the **entire sequence, prompt included**. Retrieval puts five source
passages into the prompt. So any phrase the model quotes from a source becomes a
repeated 6-gram, and the token that would finish the sentence is banned.

A model grounding its answer in the retrieved text was forbidden from finishing
the sentence it was grounding. Having no legal continuation, it stopped.

The signature is visible without any analysis. These are complete recorded
responses from `mistral-7b few_shot rag5`, cut mid-word:

> `Regressive enceph`
> `Parents and other care`
> `There is currently no one`

### The proof

`scripts/diagnose_truncation.py` rebuilds the exact prompt each response was
generated from, appends the recorded response, and replays the sequence through
the real `NoRepeatNGramLogitsProcessor` from the generation stack. It then asks
one question at the position where the response stopped: was the token that
would have continued the source passage banned there?

The rebuild is exact. **Median difference between the rebuilt prompt length and
the recorded `prompt_tokens` is 0 tokens, maximum 0, over all 208 Mistral
responses.** The sequence being replayed is the sequence the model saw.

| Run | ended inside a copied phrase | continuation banned | mean generated tokens |
|---|---|---|---|
| mistral-7b few-shot | 6% | 6% | 161 |
| mistral-7b few-shot **RAG** | **33%** | **33%** | **64** |
| mistral-7b zero-shot | 8% | 8% | 143 |
| mistral-7b zero-shot **RAG** | **35%** | **35%** | **65** |

Every response that ended inside a copied phrase ended on a banned token. Against
the human truncation grades:

| | graded complete | graded truncated |
|---|---|---|
| continuation legal | 161 | 5 |
| **continuation banned** | 2 | **40** |

**89% of truncated responses stopped on a banned token, and 95% of responses
that hit a banned token were graded truncated.** Fisher exact odds ratio 644,
p = 8.5e-35. Responses that hit the ban average 36 generated tokens against 127
for those that do not, and score **1.36 on quality against 2.03**.

That 0.67 of rubric quality was lost to a configuration flag, not to retrieval.

### The fix

`src/generation_control.py` adds `CompletionOnlyNoRepeatNGram` and
`CompletionOnlyRepetitionPenalty`. Both apply the identical constraint to the
generated text alone and ignore the prompt. `models.yaml` gains
`repetition_scope`, defaulting to `completion`; setting it to `sequence`
reproduces the original runs exactly.

The reasoning is that the degeneracy Phase 1 found
([F-P1-006](phase1-baseline.md)) is the model repeating *itself*. Repeating the
*source* is what a retrieval system is supposed to do. The old setting could not
tell the two apart. The new one can, and `tests/test_generation_control.py`
tests exactly that pair of behaviours: quoting the prompt is legal, self-repetition
is still banned.

Replaying all 208 recorded responses against the fix:

| | responses ending inside a copied phrase | continuation banned |
|---|---|---|
| `--scope sequence` (what ran) | 34% of RAG responses | **34%** |
| `--scope completion` (the fix) | 34% of RAG responses | **0%** |

Every stopping point that killed those 42 responses is legal under the fix.

### What this does not show

The replay proves the constraint blocked the continuation. It does not prove
what the model would have written instead, because that needs the weights and a
GPU. **The size of the recovery is unmeasured until the runs are repeated on
DeepNet**, and this finding claims only that the cause is identified and removed,
not that quality improves by any particular amount.

Llama could not be replayed: `meta-llama/Meta-Llama-3.1-8B-Instruct` is a gated
repository and the tokenizer will not download without credentials. Llama's RAG
responses show the same halving of length (297 to 183 tokens few-shot, 309 to 204
zero-shot) but were graded truncated 0% of the time, so the effect on Llama is
real but milder, and its size is unquantified. Re-running the diagnosis with an
`HF_TOKEN` set would close that gap.

Opens [D-111](../decisions.md), [D-112](../decisions.md).
Supersedes [D-109](../decisions.md).

---

## F-P2-011: Source recall was measuring the wrong thing

**Date.** 24 August 2026. **Artefacts.** `data/retrieval_eval/experiments.csv`,
`data/retrieval_eval/per_prompt_*.csv`. Produced by
`python scripts/evaluate_retrieval.py --label X --compare-to Y`.

### The case that exposed it

`safety_treatment` is the one category where retrieval made hallucination worse,
0% to 50% ([F-P2-009](#f-p2-009-retrieval-cuts-hallucination-by-two-thirds)). It
holds one prompt, P025, and both errors are about FDA approval:

- Llama few-shot claimed FDA approval covers "irritability, hyperactivity, and
  sleep disturbances". Approval covers irritability associated with ASD only.
- Llama zero-shot claimed the FDA "hasn't approved any medications specifically
  for treating autism symptoms". Also false, and the mirror image of the first.

**The correct answer was in the corpus and was not retrieved.**
`nichd_medication_c1` states it plainly: the FDA has approved risperidone and
aripiprazole for irritability associated with ASD. What retrieval returned was
`nichd_medication_c0`, the chunk before it, which mentions the FDA only to say
NICHD "does not endorse ... medications **not** approved by the FDA". That
sentence can be read as implying approvals exist or that none do, and the two
models read it in opposite directions.

### Why the existing metric could not see this

Source recall asks whether any chunk of the cited document appears in the top k.
For P025 the answer is yes, three times over. The document was retrieved. The
sentence was not. **A document can be forty chunks long, so retrieving the right
document is not the same as retrieving the answer**, and every retrieval number
this project has reported so far measures the first thing.

`scripts/evaluate_retrieval.py` adds **answer coverage**: the unigram recall of
the benchmark's reference answer against the retrieved context, over content
words. It is a proxy and a generous one, but it is paired across configurations
and it moves when the answer leaves the context, which source recall does not.

Current corpus, 52 prompts:

| k | source recall | answer coverage |
|---|---|---|
| 1 | 57.7% | 50.0% |
| 3 | 63.5% | 64.1% |
| 5 | 63.5% | 68.4% |
| 10 | 67.3% | 76.0% |

Source recall is flat from k=3 to k=5 while coverage keeps climbing: the extra
passages are adding answer content without adding new documents. That is the gap
the single metric was hiding.

**These figures are not comparable to `retrieval_recall_curve.csv`**, which
reports 71.2% at k=5. That curve predates the corpus curation and the source
diversity cap. The numbers here are the current pipeline, measured today.

### What fixed it

**Neighbour expansion.** Each retrieved passage brings its adjacent chunks from
the same document, inside the same k budget, so the fix trades lower-ranked
matches for the immediate context of higher-ranked ones and does not lengthen
the prompt.

| Configuration | coverage@5 | source recall@5 | MRR |
|---|---|---|---|
| baseline | 68.4% | 63.5% | 0.612 |
| **neighbour expansion** | **74.2%** | 63.5% | 0.595 |

Paired over 52 prompts: **+5.8pp coverage, p = 0.0038, better on 22 prompts and
worse on 8**. Source recall does not move at all, which is the point of the
finding: the metric the project was using is blind to the change.

MRR falls 0.018 (p = 0.026), as expected, because neighbours displace
lower-ranked hits from the cited document. That is a ranking cost with no
consequence for what the model reads.

On P025 specifically, retrieval goes from

`nichd_treatments_c2, nichd_medication_c0, nichd_treatments_c0, nichd_medication_c5, nz_..._c73`

to

`nichd_treatments_c2, nichd_treatments_c3, nichd_treatments_c1, nichd_medication_c0, **nichd_medication_c1**`

The sentence both models got wrong is now in front of them.

### What did not work, recorded because it was tried

**Stripping navigation boilerplate did nothing.** Six passages carried scraped
chrome ("Breadcrumb", "Instagram", "En Español"), all in the first chunk of a
page beside its title, and the hypothesis was that this made title chunks match
title-like queries. Measured:

| Configuration | coverage@5 | delta | p |
|---|---|---|---|
| baseline | 68.4% | — | — |
| boilerplate stripped | 68.5% | +0.1pp | 0.66 |
| neighbours | 74.2% | — | — |
| neighbours + boilerplate stripped | 74.1% | −0.0pp | 0.79 |

The hypothesis was wrong. What makes `nichd_medication_c0` match "Medication
Treatment for Autism" is the page **title**, which is real text and stays. The
strip is kept as hygiene, since nav text is not prose and should not reach a
caregiver-facing prompt, but **it is not a fix and the report should not present
it as one**. It is on by default; `--keep-boilerplate` restores the old corpus.

### Limits

Answer coverage is lexical. A retrieved passage that states the answer in
different words scores low, and a passage sharing vocabulary with the reference
answer without answering scores high. It is a screening measure for retrieval
changes, not a quality measure, and no claim in the report should rest on it
alone.

**Nothing here has been measured on generated output.** Both changes are
verified at the retrieval layer only. Whether better coverage produces better
answers needs the runs repeating on DeepNet, together with
[D-111](../decisions.md).

Opens [D-113](../decisions.md). Relates to
[F-P2-002](#f-p2-002-retrieval-is-strongest-exactly-where-the-models-hallucinated).

---

## F-P2-012: The fix holds on real output, and the rubric is still unmeasured

**Date.** 25 August 2026. **Artefacts.** `results_20260825/` (8 runs, 416
responses), `data/generation_eval/paired_truncation_fix.csv`,
`data/generation_eval/per_response_truncation_fix.csv`. Produced by
`python scripts/run_phase1.py --all --condition zero_shot few_shot` and the same
command with `--rag --top-k 5 --expand-neighbours`, then
`python scripts/evaluate_generation.py --before results_thursday --after results_20260825 --label truncation_fix`.

[F-P2-010](#f-p2-010-the-truncation-was-our-repetition-control-not-the-token-budget)
identified the cause of the truncation and verified the fix by replaying
recorded responses. This is the same fix measured on responses the models
actually generated afterwards, on the same 52 prompts, paired per prompt.

**Both arms were regenerated**, not just the retrieval arm. The repetition scope
changes baseline generation too, and 6.7% of the Phase 2 baseline responses had
also stopped on a banned token, so comparing new retrieval output against the
old baseline would have confounded the two changes.

### Responses stopped being cut off

Share of responses ending on sentence-final punctuation, paired per prompt:

| Run | Before | After | Δ | p |
|---|---|---|---|---|
| **mistral-7b few-shot RAG** | **63.5%** | **94.2%** | **+30.8pp** | 1.6e-05 |
| **mistral-7b zero-shot RAG** | **65.4%** | **94.2%** | **+28.8pp** | 0.00012 |
| llama3-8b few-shot RAG | 100% | 100% | 0 | — |
| llama3-8b zero-shot RAG | 100% | 100% | 0 | — |
| mistral-7b few-shot baseline | 92.3% | 94.2% | +1.9pp | 0.66 |
| mistral-7b zero-shot baseline | 90.4% | 94.2% | +3.8pp | 0.42 |
| llama3-8b few-shot baseline | 96.2% | 96.2% | 0 | 1.00 |
| llama3-8b zero-shot baseline | 92.3% | 94.2% | +1.9pp | 0.71 |

**The two runs the diagnosis predicted would change are the two that changed**,
and the six it predicted would not are flat. Pooled over the retrieval arm,
82.2% to 97.1%.

The completeness measure here is automatic: a response counts as complete if it
ends on sentence-final punctuation. **It agrees with the human truncation grades
on 99.3% of the 416 graded Phase 2 responses**, which is why it is reported, but
it is still a proxy and the report should call it one.

### The recovered length is where the constraint was biting

| Run | Before | After | Δ tokens | p |
|---|---|---|---|---|
| mistral-7b zero-shot RAG | 64.9 | 127.8 | **+62.9** | 4.8e-08 |
| mistral-7b few-shot RAG | 64.4 | 108.3 | **+44.0** | 3.3e-06 |
| mistral-7b zero-shot baseline | 143.2 | 168.0 | +24.8 | 0.027 |
| mistral-7b few-shot baseline | 161.0 | 178.7 | +17.7 | 0.25 |
| llama3-8b few-shot RAG | 182.5 | 186.5 | +4.0 | 0.71 |
| llama3-8b zero-shot RAG | 203.5 | 202.0 | −1.5 | 0.91 |
| llama3-8b few-shot baseline | 297.5 | 265.8 | −31.7 | 0.018 |
| llama3-8b zero-shot baseline | 309.4 | 261.8 | −47.6 | 0.011 |

`p2_length_recovery.png` shows the shape rather than the mean: the spike of
Mistral retrieval responses under 50 tokens, which is what a banned continuation
produces, is gone.

**Llama's baseline responses got shorter, by 32 and 48 tokens, both significant.**
This is not explained by the truncation fix, since Llama was never truncated and
its completeness did not change. The likely cause is the repetition penalty no
longer applying to prompt tokens, which removes pressure to keep introducing new
vocabulary, but that is a hypothesis and it has not been tested. Shorter is not
worse on its own, and the rubric will say whether it matters. Recorded as
[D-115](../decisions.md).

### Degeneracy did not return

**0 of 416 responses degenerate**, matching the Phase 1 result
([F-P1-006](phase1-baseline.md)). This is the check that mattered for the fix:
the repetition control was scoped, not removed, and the protection it exists for
still holds.

### What this does not show

**No hallucination or quality figure in this finding.** Those are rubric
measures, they require the blind grading pass, and these 416 responses have not
been graded. Every number above is mechanical.

That matters most for the two changes whose whole purpose was accuracy.
Neighbour expansion was adopted because it raises answer coverage
([F-P2-011](#f-p2-011-source-recall-was-measuring-the-wrong-thing)), and the
grounded prompt was changed to restore follow-up advice
([D-114](../decisions.md)); **neither claim is supported by anything here**. The
retrieval arm of this run set differs from Phase 2 in three ways at once —
repetition scope, neighbour expansion, and the prompt — so even after grading,
the change from Phase 2 cannot be attributed to any one of them without further
ablation. Recorded as [D-116](../decisions.md).

Partially resolves [D-111](../decisions.md): the truncation half is answered,
the rubric half is not.

---

## F-P2-013: The re-run graded, and what retrieval actually fixes

**Date.** 25 August 2026. **Artefacts.** `data/grading_20260825/` (416
annotations in 16 batches, `rerun_results_scored.csv`,
`rerun_rubric_paired.csv`), tables `rerun_rubric_by_run`,
`rerun_rubric_paired`, `rerun_error_families`. Produced by
`python scripts/prepare_grading.py --results-dir results_20260825 --grading-dir data/grading_20260825 --seed 20260825`,
sixteen batches of manual annotation, then
`python scripts/score_rag_rubric.py --grading-dir data/grading_20260825 --prefix rerun`.

All 416 responses from `results_20260825/` graded blind to model, condition and
retrieval status, shuffled across all eight runs, with the mapping opened only
after the last batch. This answers [D-111](../decisions.md) and
[D-113](../decisions.md), and it is the first rubric evidence for neighbour
expansion and the grounded prompt change.

### Headline

| Measure | Baseline | RAG | Δ | p | dz |
|---|---|---|---|---|---|
| **Hallucination** | **30.3%** | **11.1%** | **−19.2pp** | 1.7e-07 | −0.38 |
| **Quality (0-3)** | **1.889** | **2.375** | **+0.486** | 1.2e-09 | 0.44 |
| Diagnostic overreach | 0.0% | 0.0% | 0 | — | — |
| Answers the question | 91.3% | 91.8% | +0.5pp | — | — |
| Recommends professional follow-up | 32.7% | 33.2% | +0.5pp | — | — |
| Truncated mid-sentence | 0.0% | 0.0% | 0 | — | — |

Paired by prompt, 208 pairs. **Both headline effects hold in all four runs
separately**: hallucination at p = 0.011, 0.011, 0.0019, 0.028, and quality at
p = 0.0016, 6.5e-05, 0.019, 0.013.

### The three fixes did what they were meant to

Against [F-P2-009](#f-p2-009-retrieval-cuts-hallucination-by-two-thirds), which
graded the same benchmark before any of them:

| Measure, retrieval effect | Phase 2 | After the fixes |
|---|---|---|
| Hallucination | −26.9pp | −19.2pp |
| Quality | +0.16 | **+0.49** |
| Answers the question | **−10.6pp** | **+0.5pp** |
| Professional follow-up | **−9.1pp** | **+0.5pp** |
| Truncated | +11.1pp | 0 |

**The two measures that moved the wrong way no longer do.** The
answers-the-question loss was the truncation fault and is gone with it. The
follow-up advice loss was the sources instruction displacing the caregiver
instructions, and restating them after the sources block reversed it. That
resolves [D-110](../decisions.md) and supports [D-114](../decisions.md).

**`safety_treatment` went from 50% to 0%.** That was the single category where
retrieval made hallucination worse, and the reason
[F-P2-011](#f-p2-011-source-recall-was-measuring-the-wrong-thing) added
neighbour expansion. Both medication responses in this pass state the FDA
approval of risperidone and aripiprazole for irritability correctly, which no
retrieval response managed in Phase 2.

### The honest reading of the headline

**The retrieval arm's hallucination rate did not improve: it is 11.1% in both
passes, to three figures.** What changed is the baseline, from 38.0% to 30.3%.
So the measured *effect of retrieval* is smaller than Phase 2 reported, and the
report should say the smaller number, −19.2pp, not the larger one.

Why the baseline improved is not established. The generation fix touched both
arms and 6.7% of Phase 2 baseline responses had stopped on a banned token
([F-P2-010](#f-p2-010-the-truncation-was-our-repetition-control-not-the-token-budget)),
but that is too small to explain 7.7pp on its own, and sampling variance across
a fresh run is uncontrolled. Recorded as [D-117](../decisions.md).

Quality is the clearer gain: the retrieval arm rose from 2.139 to 2.375 while
the baseline fell slightly, so the effect size more than tripled.

### What retrieval fixes, and what it does not

`rerun_error_families` classifies every graded response by the annotator's own
note and splits it by arm. This is the most useful result in the pass.

| Error family | Baseline | RAG |
|---|---|---|
| Advises correcting for prematurity | 4 | **0** |
| Says the M-CHAT needs a trained administrator | 4 | **0** |
| Advises watchful waiting on a positive screen | 3 | **0** |
| Stale "1 in 44" prevalence figure | 11 | **1** |
| Fabricated citation or instrument | 15 | 13 |
| Refusal or non-answer | 18 | 17 |
| Denies the autism-encephalopathy link | 4 | 4 |

**Every response that misused a screening instrument was a baseline response.**
Retrieval eliminated three families outright and all but one instance of the
stale prevalence figure. These are exactly the errors a corpus can fix: a
specific fact, stated in a source, that the model otherwise recalls wrongly.

**Fabricated citations are not a retrieval phenomenon**, splitting 15 to 13.
Neither are refusals, 18 to 17. Retrieval does not stop a model inventing a
"2019 Journal of Pediatrics study", a "Ritvo-Folstein scale" or "MitoCHAND
syndrome", and it does not stop it declining to answer. **This is the strongest
argument in the project for why retrieval is necessary but not sufficient**, and
it is the natural motivation for the Phase 3 routing layer.

The autism-encephalopathy denial splits 4 to 4, which fits
[F-P2-011](#f-p2-011-source-recall-was-measuring-the-wrong-thing): retrieval only
helps when it surfaces the passage, and `general_health_knowledge` remains the
worst-retrieved category at 35% hallucination in both arms.

### What still holds

**Diagnostic overreach is zero across all 416 responses**, in both arms. That is
1,008 graded responses across three passes with none crossing the diagnosis
line.

Two responses scored 0 for being actively misleading: G0045, a retrieval
response telling a caregiver adults cannot be diagnosed with autism, and G0280,
a baseline response concluding from a prevalence figure that one of two siblings
is likely autistic. One in each arm.

### Limits

Grades are model-assisted with human adjudication and there is still no
independent second scorer ([D-105](../decisions.md)). **The retrieval arm
changed in three ways at once** — repetition scope, neighbour expansion, the
prompt — so no single change can be credited from this data alone
([D-116](../decisions.md)); the category-level and error-family evidence above
is suggestive but not an ablation. Four adjudication questions from this pass
remain open, listed in the grading log.

Resolves [D-110](../decisions.md), [D-111](../decisions.md),
[D-113](../decisions.md). Opens [D-117](../decisions.md).

### Addendum, 25 August 2026: which comparison the caveat applies to

The section above can be misread as qualifying the headline. It does not, and
the distinction is worth stating plainly because the report depends on it.

**Within this pass, baseline against retrieval is a clean paired comparison.**
Both arms were generated in the same session under identical settings, answer
the same 52 prompts, and were graded blind in the same pass by the same
annotator. 30.3% to 11.1% at p = 1.7e-07 is the result, and it is not
provisional in any respect.

**The caveat is only about comparing this pass's delta with the July one.**
−19.2pp here against −26.9pp there differ because the baseline arm differs
between passes, so the two deltas cannot be quoted interchangeably or shown as
a trend. The re-run is the experiment; the July grading is the earlier state.

**Phase 1 is a separate experiment again**, on 44 prompts rather than 52, and
its rates do not belong on an axis with either.

---

## F-P2-014: The automated metrics agree that the generation fix worked

**Date.** 26 August 2026. **Artefacts.** `results_20260825/*_with_metrics.csv`.
Produced by `python scripts/run_phase1.py --metrics <path>` for each of the eight
runs.

The 25 August runs were graded before they were scored: they had no BERTScore or
ROUGE-L at all, so the layer-by-layer comparison had a hole in it. That is now
closed, and the numbers say something the rubric could not.

### The scorer is now pinned and recorded

`bert_score` resolves `lang="en"` to `roberta-large`, but nothing in the
repository recorded which encoder produced the July numbers, and a BERTScore is
only comparable against another computed with the same encoder.
`src/metrics.py` now pins `BERTSCORE_MODEL` explicitly and writes a
`bertscore_scorer` column carrying the encoder and the `bert_score` version on
every row. Pinning to `roberta-large` keeps the July figures comparable while
making the choice visible from here on.

The pin reproduces: `mistral-7b zero-shot` baseline scores **0.861 / 0.176**
today against **0.861 / 0.171** in July, on the same 52 prompts.

### Retrieval, on the automated metrics

| | Baseline | With retrieval |
|---|---|---|
| BERTScore F1 | 0.853 | **0.886** |
| ROUGE-L | 0.158 | **0.318** |

### The fix is visible to ROUGE-L, and only in the arm it touched

| Run | ROUGE-L July | ROUGE-L August |
|---|---|---|
| mistral-7b few-shot **RAG** | 0.207 | **0.362** |
| mistral-7b zero-shot **RAG** | 0.195 | **0.357** |
| llama3-8b few-shot **RAG** | 0.225 | 0.286 |
| llama3-8b zero-shot **RAG** | 0.224 | 0.265 |
| mistral-7b few-shot baseline | 0.170 | 0.177 |
| mistral-7b zero-shot baseline | 0.171 | 0.176 |
| llama3-8b few-shot baseline | 0.140 | 0.142 |
| llama3-8b zero-shot baseline | 0.132 | 0.137 |

**Mistral's retrieval ROUGE-L nearly doubles while its baseline moves by 0.006.**
That is the signature of the truncation fix
([F-P2-010](#f-p2-010-the-truncation-was-our-repetition-control-not-the-token-budget)):
a response cut off after 40 tokens cannot overlap a reference answer, and Mistral
retrieval was the only arm being cut off. The automated metrics were never told
about that fault and had no part in diagnosing it, so this is independent
corroboration rather than a restatement.

### What this says about D-117, and what it does not

All four **baseline** runs are stable across the two passes, moving by at most
0.005 BERTScore and 0.007 ROUGE-L. That is a constraint on
[D-117](../decisions.md), which records an unexplained 7.7pp fall in baseline
hallucination between the same two passes.

**It is a weak constraint and should not be quoted as resolving anything.**
[F-P1-005](phase1-baseline.md) established on this project's own data that
automated similarity does not track hallucination. An arm can therefore be stable
on BERTScore and still differ on factual accuracy, so the stability above is
consistent both with the rubric difference being real and with it being grading
variance. D-117 stays open, and the seeded regeneration it calls for is the thing
that would settle it.

### Limits

The July runs carry no `bertscore_scorer` column, because the pin did not exist
when they were scored. They are believed to be `roberta-large` because that is
what `lang="en"` resolved to then and the reproduced figure above matches to
three decimal places, but that is inference, not a record.

### Addendum, 26 August 2026: the reconstruction no longer reproduces exactly

F-P2-010 reports "median difference between the rebuilt prompt length and the
recorded `prompt_tokens` is 0 tokens, maximum 0, over all 208 Mistral responses",
and offers that as evidence the replayed sequence was the sequence the model saw.
That was true when it was written and is no longer reproducible.

`build_grounded_system_prompt` gained the D-114 restatement on 25 August, after
those July runs were generated. Replaying them through today's builder therefore
adds roughly fifty tokens to every retrieval prompt, and the same command now
reports a median difference of 26 and a maximum of 52.

**The finding is unaffected.** Re-running gives the same blocked-continuation
figures, 34% of retrieval responses and 7% of baseline responses, because whether
a response's final n-gram also appears in the retrieved sources does not depend on
an instruction paragraph added above them. What is no longer true is the exactness
claim, which belongs to the builder as it stood on 25 August rather than to the
current one. `scripts/diagnose_truncation.py` now says so in its docstring.

Recorded rather than quietly corrected, because a number in a note that does not
reproduce is exactly the failure the working agreement is meant to catch.

---

## F-P2-015: Retrieval did not reduce follow-up advice; it stopped aiming it

**Date:** 26 August 2026
**Artefacts:** `data/generation_eval/followup_conformance_rerun_20260825.csv`,
`followup_paired_rerun_20260825.csv`,
`followup_discrimination_rerun_20260825.csv`
**Command:** `python scripts/evaluate_followup.py --scored
data/grading_20260825/rerun_results_scored.csv --label rerun_20260825`
**Figure:** `docs/figures/05_phase2_rerun_aug/followup_targeting_collapse.png`

The concern this set out to check was that retrieval had reduced professional
follow-up advice. On the flat rate it had not: baseline 32.7%, retrieval 33.2%,
across all 208 responses per arm. On that measure nothing happened at all.

The flat rate was the wrong measure. It counts a caregiver asking what
prevalence means and a caregiver asking where to get their child assessed as the
same kind of turn. `config/prompts.yaml` declares per route whether a next step
is `required`, `expected` or `optional`, frozen before any routed output
existed, and scoring against those obligations separates two things the flat
rate merges: **conformance**, the share of turns that owed a next step and got
one, and the **unprompted rate**, the share of turns that owed nothing and got
one anyway. Their difference is how well an arm aims its advice.

| Arm | Flat rate | Owed a next step | Owed nothing | Targeting gap | p |
|---|---|---|---|---|---|
| Baseline | 32.7% | 52.5% | 28.0% | **+24.5pp** | 0.0068 |
| Retrieval | 33.2% | 32.5% | 33.3% | **−0.8pp** | 0.921 |

**The baseline aims its advice and retrieval does not.** The baseline advises
the turns that owe a next step half again as often as the turns that do not.
Under retrieval the two are indistinguishable: it advises everything at the same
rate, which is what a system with no sense of what the turn needs looks like.
The gap closes by 25.4pp, 95% CI [−42.4, −7.7], bootstrapped over the 52 prompts
rather than the 416 responses because four responses to one prompt are not
independent and prompts are what is scarce. **That interval excludes zero.**

The per-bucket paired comparisons do not, and it matters that they are reported
that way:

| Bucket | n pairs | Baseline | Retrieval | Difference 95% CI | p |
|---|---|---|---|---|---|
| Owes a next step | 40 | 52.5% | 32.5% | −20.0pp [−42.0, +2.0] | 0.073 |
| required | 28 | 42.9% | 28.6% | −14.3pp [−41.6, +13.1] | 0.293 |
| expected | 12 | 75.0% | 41.7% | −33.3pp [−74.7, +8.1] | 0.104 |
| optional | 168 | 28.0% | 33.3% | +5.4pp [−4.1, +14.8] | 0.266 |

Every direction is consistent — less advice where it is owed, slightly more
where it is not — and not one bucket is significant on its own. The supported
claim is the interaction, not any single row. Reporting the rows without their
intervals would overstate this considerably.

**What this cannot show, and it is a hard limit.** Only 10 of the 52 benchmark
prompts owe a next step: 7 `required` and 3 `expected`. Every conformance figure
above rests on those ten, and `expected` rests on three. The benchmark was
written to test factual grounding and was never designed to exercise follow-up
obligations. A conformance measure this underpowered can support the interaction
and cannot support a per-route reading, and the report must say so wherever the
number appears.

**The detector measures presence, not appropriateness.** `FOLLOWUP_PATTERNS` in
`src/rubric.py` matches phrases like "speak to your doctor". A response naming
the wrong service, or burying the advice, or giving it for the wrong reason,
scores identically to one that gets it right. Conformance is a floor on quality,
never a measure of it. The route-conditioned arm was built partly to address
this, and its `referral` and `result_explanation` blocks ask for a next step
concrete enough to act on today — but nothing here measures concreteness.

**Why this is worth acting on rather than noting.** The obvious fix is to push
the flat rate up, and this finding says that would be the wrong fix: retrieval
already advises the optional turns slightly *more* than baseline does, and a
flat-rate target would push it further in the direction that is already wrong.
What the routed arm has to demonstrate is a restored gap, not a raised rate.
That prediction is on record before the routed arm has been generated.

---

### Addendum to F-P2-012, 26 August 2026: the similarity metrics corroborate the truncation fix

Not a new finding, and it does not change F-P2-012. Recorded because the numbers
table it cites gained sixteen rows and a reader comparing versions should know
why.

`evaluate_generation.py` became N-way to serve the end-to-end ladder, and in the
process it began reading the `*_with_metrics.csv` sidecars rather than the plain
run files. Those sidecars did not exist when the July-against-August comparison
was first run; they were written when the metrics gap was closed on 26 August.
So the same comparison now carries BERTScore F1 and ROUGE-L alongside
completeness, and `docs/report/numbers/p2_generation_fix.csv` has the rows.

Every previously published row is unchanged, and
`tests/test_generation_comparison.py` pins that against a frozen copy of the
original output in `tests/fixtures/`.

What the new rows say, from the artefact: the fix moved both similarity metrics
on all four retrieval runs and on essentially none of the four baseline runs.
Every retrieval run improves at p < 0.05 on both metrics; among the baseline
runs, one of four reaches significance on BERTScore and none on ROUGE-L. The
largest movement is Mistral zero-shot RAG on ROUGE-L.

That is the pattern the truncation explanation predicts and a generation-budget
explanation does not. The baseline responses were never being cut off, so a fix
to the repetition constraint should leave them where they were, and it did. It
is weak corroboration — these are automated similarity metrics, which this
project treats as descriptive only and on which no decision rests — but it is
corroboration from a measure that was not used to reach the original conclusion.
