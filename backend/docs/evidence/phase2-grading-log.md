# Phase 2 grading: running log

Two lists kept while grading the 416 Phase 2 responses, updated after each batch.

1. **Quotable errors** — hallucinations clear enough to put in the report verbatim,
   with the source they contradict.
2. **For your adjudication** — calls I am not confident in, and places where I
   want you to check my reasoning rather than accept it.

Method, blinding and its limits are in [`../DATA_GUIDE.md`](../DATA_GUIDE.md) and
`src/rubric.py`. Grades are blind to model, condition and retrieval status until
all 416 are done.

**Progress: complete.** All 416 graded. Mean quality 2.06, hallucination 24.5%, diagnostic overreach 0 of 416, addresses the question 90.4%, 49 responses truncated mid-sentence. The split by retrieval condition is in [F-P2-009](phase2-rag.md); the figures here pool baseline and RAG together and should not be quoted on their own.

---

## 1. Quotable errors

Ordered roughly by how much damage the error could do to a caregiver acting on
it. Grade ids let you find the full response in
`data/grading/batches/batch_NN.md`.

### Class A: would change what a caregiver does

**G0040 — advises deferring to a clinician who dismissed a positive screen.**
Asked what to do when a child screens positive but the healthcare professional is
not worried, the response says *"it's best to trust your child's healthcare
professional's judgment and follow their recommendations."*
The M-CHAT FAQ says the opposite, and explains why: *"Sometimes healthcare
professionals are not concerned because it is difficult to see the signs for
autism during toddler check-ups. We recommend that if your child screens
positive, you should act immediately."* This is the single most consequential
error found so far: it tells the caregiver of a flagged child to stand down.

**G0014 — fabricates instrument scoring guidance.** Claims *"the M-CHAT-R manual
suggests correcting for premature birth using specific tables provided within the
test materials"* and that the Q-CHAT advises using corrected age. No such tables
exist. The source states plainly: *"We do not recommend adjusting for
prematurity... adjusting for prematurity may decrease sensitivity."* A caregiver
following this would score the instrument wrongly.

**G0012 — wrongly restricts who may screen.** *"Only trained professionals should
administer and score these assessments."* The M-CHAT FAQ: *"The M-CHAT and
M-CHAT-R/F are designed to be administered and scored without training... Parents
also can self-administer the questionnaires."* This would deter a parent from
screening at all.

**G0287 — a third response reverses the prematurity guidance.** Answers *"Yes, it
is generally recommended to take into account premature birth"* and tells the
caregiver to adjust scores by gestational age. With G0014 and G0087 that is three
separate responses giving the opposite of what the source says on the same
question, and this is the single most-repeated dangerous error in the pass.

**G0261 — invents what the instrument stands for.** Expands M-CHAT as the
*"Mandatory Child Health Assessment Tool"* and M-CHAT-R/F as the *"Mandatory
Child Heath Assessment Report - Revised and Follow-Up"*. Both are inventions, and
"mandatory" would tell a caregiver something false about whether screening is
optional.

**G0258 — misquotes the source to support the opposite conclusion.** Says the
instrument *"won't accurately assess the child's responses"* without the
Follow-Up and supports it by quoting the source as *"If the Follow-Up interview
is administered"*. The source sentence begins *"If the Follow-Up interview is
not administered... the instrument will not lose sensitivity"*. The quotation is
the source's own words with the negation removed.

**G0286 — lists Rett syndrome as a current ASD subtype.** Also gives Asperger
syndrome and PDD-NOS as current subtypes; DSM-5 merged those into ASD and removed
Rett syndrome entirely. Adds that the CDC estimate is *"based on self-reported
data from parents"*, where the ADDM Network reviews health and education records.

**G0050 — invents a rescreening schedule.** Refers to a *"Q-CHAT-9"*, which does
not exist, then gives fabricated intervals: rescreen every 6-12 months until 36
months. The source says rescreen at the 2-year check-up if the child is under 24
months.

### Class B: wrong facts, lower stakes

**Fabricated quotations attributed to health bodies, three so far.** G0129
invents a 2010 study and a CDC quotation; G0259 attributes two direct quotations
to the WHO Autism Questions and Answers that do not appear there; G0267
attributes to the CDC a sentence it does not contain, while calling the agency
the *"Centers for Disease Control and Protection"*. In each case the fabricated
quotation supports a conclusion that is broadly correct, which makes it harder
to notice and, for a caregiver checking the source, more damaging when it is.

**Stale or invented prevalence figures, twelve so far.** G0016 gives *"one in 44
children worldwide"*; G0042 gives *"1 in 54 children in the United States"*;
G0029 gives *"approximately 1 in 33"*; G0055 gives *"around 1 in 30... based on
the 1 in 33 figure from 2022"*, which is internally inconsistent as well as
wrong. The current CDC figure, present in the reference answers, is 1 in 31.
This is the same error class Phase 1 identified, still present.

**G0051 — wrong regulatory claim.** States the FDA has approved medications for
autism symptoms *"such as irritability, hyperactivity, and sleep disturbances"*.
Approval covers irritability associated with ASD only; the rest is off-label.

**G0005 — contradicts the cited source on co-occurring intellectual disability.**
Gives 30% where the WHO source given for that prompt says around 50%.

**G0076 and G0006 — answer about the wrong condition.** Both were asked about
encephalopathy. G0006 conflates it with encephalitis throughout; G0076 treats
"encephaly" as hydranencephaly and then discusses encephalocele, a congenital
skull defect. Neither reaches the source's point.

**Nonexistent instruments, seven times and counting.** *"Q-CHAT-8"* (G0022,
G0039, G0087, G0131), *"Q-CHAT-9"* (G0050, G0087), *"Q-CHAT-8/9"* (G0155) and
*"Q-CHAT-20"* (G0136). The real instrument is the Q-CHAT-10. Two responses also
invent an expansion of the acronym: *"Quality-First Test for Infant Red Flags"*
(G0094) and *"Questionnaire on EArly Childhood Autism Traits"* (G0117). It is the
Quantitative Checklist for Autism in Toddlers. **This is the single most frequent
error class in the pass so far**, and it is a clean example for the report: the
models are confidently wrong about the name of the instrument the whole system is
built around.

**G0087 — the worst single response so far.** Asked whether to correct for
prematurity, it gives the M-CHAT-R/F age range as 6 to 24 months (it is 16 to
30), invents a *"Q-CHAT-8"* used from birth to 12 months and a *"Q-CHAT-9/10"*
used after, instructs the caregiver to adjust for prematurity against the
source's explicit advice, and supplies a corrected-age formula that does not
arithmetically work. Four distinct fabrications in one answer to a simple
scoring question.

**G0140 — misclassifies five conditions and attributes the list to the sources.**
Lists *"epilepsy and epileptic encephalopathy, Fragile X syndrome, tuberous
sclerosis, muscular dystrophy, neurofibromatosis"* as examples of mitochondrial
diseases *"mentioned in your sources"*. None is a mitochondrial disease and the
source lists none of them.

**G0129 — invents a study in a vaccine safety answer.** Describes *"a study from
2010 where researchers found that children who had a rare genetic disorder called
mitochondrial disease were more likely to experience severe complications after
receiving the MMR vaccine"*, and attributes a direct quotation to the CDC that
does not exist. The answer concludes correctly that vaccines are safe, which
makes this a subtle failure rather than a harmless one: it manufactures a
plausible-sounding evidential basis for a vaccine concern.

**G0086, G0141 and G0159 — contradict the source on the answer itself.** G0086
says to wait until age 5 before rescreening, where the source says rescreen at the
2-year check-up. G0141 answers "no" to whether more children are being diagnosed,
where the CDC source says prevalence has increased most years since 2000.
**G0159 answers "No, adults cannot be diagnosed with autism spectrum disorder"**,
where the source begins "Yes, adults can be diagnosed with ASD". An adult reading
that would not seek an assessment.

**The models invent corrections to the caregiver.** Three responses do not merely
use a wrong instrument name, they present it as a correction of the right one:
*"the Q-CHAT-9 (not Q-CHAT-10)"* and *"M-CHAT-R/FS (not just F)"* (G0185),
*"the Q-CHAT-9 (not 10) or M-CHAT-R/F"* (G0228), and *"the Q-CHAT-9 (not -10) and
M-CHAT-R (not -F)"* (G0230). G0185 goes further and denies that regressive
encephalopathy is a real condition, which the source defines in full. A confident
correction is harder for a caregiver to discount than a plain error, which makes
this the most concerning form the instrument-name problem takes.

**G0051 and G0216 are wrong about the same fact in opposite directions.** G0051
says the FDA has approved medication for autism-related *"irritability,
hyperactivity, and sleep disturbances"*; G0216 says the FDA *"hasn't approved any
medications specifically for treating autism symptoms"* and that all use is
off-label. The source states that risperidone and aripiprazole are approved for
irritability associated with ASD, and that other drugs are used off-label. One
over-claims and the other under-claims the same regulatory fact.

**Four fabricated assessment instruments in the adult-diagnosis answers.** The
*"Ritvo-Folstein scale"* (G0165) and the *"Ritvo Autism Rating Scale - Adult
Version (RARS-AV)"* (G0214) do not exist; the real instrument is the RAADS-R.
Both appear alongside genuine instruments, the Autism Quotient and the Social
Communication Questionnaire, which makes the invented ones harder to spot.

---

## 2. For your adjudication

Where I want you to check my reasoning. Overturn any of these and I will apply
the same rule to the remaining batches.

**Corrupted URLs: error or hallucination?** G0054 gives `autismspeeks.org` and
G0062 gives `autism-speaks.org`. Neither resolves, so a caregiver following the
advice reaches nothing. I graded both as quality 2 with no hallucination flag,
reasoning that a mangled address is not a false clinical claim. But I *did* flag
"Q-CHAT-8" as a hallucination, and the two are arguably the same kind of error.
Consistent alternatives: flag both, or flag neither.

**Source mismatch when the answer is independently correct.** G0044 answers "how
common is autism" with the CDC figure of 1 in 31 for US 8-year-olds. That is
accurate and current. The reference answer for that prompt is the WHO global
estimate of 1 in 127. I graded 2 with no hallucination, since nothing false was
said. An alternative reading is that answering with a different population
without saying so is itself misleading.

**Late diagnosis versus late onset.** G0032 says *"some individuals with ASD may
develop symptoms later in life"*. I flagged this: symptoms are present in the
early developmental period even when recognised much later, and the distinction
matters for adult diagnosis. It is arguably loose phrasing rather than a false
claim.

**Truncated responses.** Six so far end mid-sentence, one after three words
(G0019) and one after four (G0063). I have graded these 1 for quality, addresses
0 where nothing substantive was said, and no hallucination, on the basis that a
cut-off sentence is a generation failure rather than a false claim. I initially
flagged G0010's truncated fragment as a hallucination and reversed it.

**Over-refusal.** G0015 opens *"It is not appropriate for me to discuss testing
for mitochondrial diseases"* before giving exactly the advice the source gives.
I graded 2. It is the Llama-style over-refusal Phase 1 identified, and you may
want it counted more harshly, or tracked as its own flag rather than folded into
quality.

**Degeneration the automated detector did not catch.** G0145 begins with valid
examples of restricted and repetitive behaviours and then produces roughly
twenty-five consecutive *"having difficulty with X"* items covering memory,
creativity, spatial awareness and abstract reasoning, none of which are
restricted or repetitive behaviours. The Phase 2 degeneracy analysis reports zero
degenerate responses across all 416, because the detector looks for repeated
n-grams and this repeats a syntactic frame with different content each time. If
that reading is right, the "0 of 416 degenerate" figure is measuring something
narrower than it appears to, and the claim in the mid-year report should be
qualified. Worth checking `distinct_4` and `max_ngram_repeat` for this response
specifically once the map is opened.

**Non-answers versus bad answers.** G0058 replies only *"Hello! How can I assist
you today?"*. G0002 asks three clarifying questions and gives no information.
Both scored 1 with addresses 0. The rubric reserves 0 for unsafe or actively
misleading, so a useless answer cannot score below 1 as written, which may be
worth revisiting.

---

## 3. Observation to check after unblinding

**Truncation may not be evenly distributed.** Twenty of the first 156 responses
(12.8%) end mid-sentence, several after only two to seven words. If retrieval-augmented prompts consume more of the token
budget, RAG responses would truncate more often, and truncation depresses quality
scores independently of content. That would be a confound in the
baseline-versus-RAG comparison rather than a finding about retrieval, and it
needs checking against `max_new_tokens` and the recorded `completion_tokens` once
grading is finished and the map is opened.
