# Final report outline

The argument the report makes, section by section, and which recorded finding
supports each claim. Update as the work develops. Findings are cited by id; the
numbers behind them are in [`NUMBERS.md`](NUMBERS.md) and explained in
[`../DATA_GUIDE.md`](../DATA_GUIDE.md).

## The one-sentence argument

A general-purpose LLM is already safe enough at refusing to diagnose but not
accurate enough about the screening instruments to be trusted with a caregiver,
and the fix is not a better model but better engineering around it: sourced
retrieval for accuracy, and an inspectable router for safety-critical dispatch.

That sentence is what every section has to earn.

## The three questions

| | Question | Phase | State |
|---|---|---|---|
| RQ1 | Can a plain LLM help a caregiver through autism screening? | 1 | Answered. Safe at the boundary, unreliable on instrument facts. |
| RQ2 | Does grounding it in clinical sources stop the factual errors? | 2 | Half answered. Retrieval works and moves output; the safety effect is unmeasured. |
| RQ3 | Can an agentic layer route caregiver turns safely? | 3 | In progress. Measurement fixed, router itself still weak. |

## Section plan

### Introduction
Screening is not diagnosis, and the gap between them is where caregivers get
stranded. The Autism AI classifier returns 0 or 1 with no explanation; the
research is the conversational layer around it, not the classifier. State the
three questions and the non-negotiable constraint: never diagnose.

### Literature Review
Four threads: LLMs in clinical and health communication; retrieval-augmented
generation for factual grounding; agentic and multi-agent architectures,
especially the argument for hybrid symbolic-neural designs in safety-critical
domains; and autism screening instruments (Q-CHAT-10, M-CHAT-R/F) with their
validation literature. See [`../references.md`](../references.md).

### Methodology
System architecture; the benchmark and its three versions, with the reference
answers quoted from primary sources; the automated metrics and why they are
secondary; the rubric and its five flags; the retrieval pipeline; the router and
its evaluation protocol. The protocol subsection is where the repeated
cross-validation argument belongs — see [F-P3-003](../evidence/phase3-router.md).

### Phase 1 results
F-P1-001 (safety held) → F-P1-002 (accuracy did not) → F-P1-003 (and the errors
are specifically about the instruments) → F-P1-004 (the two models fail in
opposite ways) → F-P1-005 (and the automated metrics could not see any of it) →
F-P1-006 (nor could they see degeneracy).

The narrative order matters: the reassuring result first, then the problem, then
the demonstration that the standard metrics would have missed the problem. That
last step is what motivates everything after it.

### Phase 2 results
F-P2-001 (retrieval baseline) → F-P2-002 (strongest where Phase 1 was weakest) →
F-P2-003 and F-P2-004 (the two corpus decisions, both measured) → F-P2-005 (source
diversity) → F-P2-007 (generation results, and convergence between models) →
F-P2-008 (what this cannot yet claim).

The corpus engineering is under-reported and worth real space: a smaller corpus
retrieving better is a genuine finding, not housekeeping.

### Phase 3 results
F-P3-001 and F-P3-002 (first router, and the claim made for it) → F-P3-003 (what
happened when it was measured properly) → F-P3-004 (what the rule layer is worth)
→ F-P3-005 (the taxonomy split, re-measured) → F-P3-006 (a fifth of the errors
are label disagreements) → whatever the improvement work produces.

This is the section where the progression narrative is the contribution. Each
step should read: what we believed, what we changed, what it measured, what that
told us.

### Discussion
Three arguments. Automated similarity metrics are inadequate for safety-critical
generation, with our own data as the demonstration. Engineering the corpus beat
engineering the model. And measurement discipline is itself a finding: two of our
own earlier claims did not survive being measured properly, which is worth
saying out loud rather than quietly correcting.

### Conclusions and Future Work
Rubric grading of the RAG outputs is the largest outstanding gap. Then: end-to-end
agent evaluation rather than routing accuracy as a proxy; a second rater; and
embedding-based routing once the data supports it.

## Where the sections stand

| Section | Material | Missing |
|---|---|---|
| Introduction | Mid-year §1 | Scope and objectives, excluded from the mid-year report |
| Literature Review | Literature review submission, April 2026 | Needs updating for the agentic and RAG threads |
| Methodology | Mid-year §2, §4 | Rubric definition, router protocol |
| Phase 1 | Complete | Second rater |
| Phase 2 | Retrieval and generation complete | Rubric grading of RAG outputs |
| Phase 3 | In progress | The improvement work itself |
| Discussion | Arguments identified | Written once Phase 3 concludes |
