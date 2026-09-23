# Re-run grading: running log

Two lists kept while grading the 416 responses in `results_20260825/`, the same
shape as [`phase2-grading-log.md`](phase2-grading-log.md) for the July pass.

1. **Quotable errors** — false claims clear enough to put in the report
   verbatim, with the source they contradict.
2. **For your adjudication** — calls I am not confident in, where I want the
   reasoning checked rather than accepted.

Method and blinding are in [`../DATA_GUIDE.md`](../DATA_GUIDE.md) and
`src/rubric.py`. Grades were blind to model, condition and retrieval status
until all 416 were finished. The arm attributions below were added after
unblinding.

**Progress: complete.** Pooled across both arms, which is not a quotable
figure on its own: mean quality 2.132, hallucination 20.7%, addresses the
question 91.6%, diagnostic overreach 0 of 416. The split by arm is
[F-P2-013](phase2-rag.md).

---

## 1. Quotable errors

Ordered by how much damage the error could do to a caregiver acting on it.

### Actively misleading

**G0280** (baseline, llama3-8b zero-shot). Asked how common autism is, gave the
superseded 1 in 44 figure and then concluded: *"if you have two children, it's
likely that one of them will be on the autism spectrum."* Off by roughly a
factor of twenty. A caregiver could read it as meaning a sibling is probably
autistic. Scored 0.

**G0045** (retrieval, mistral-7b zero-shot). Asked whether adults can be
diagnosed with ASD, answered *"No, adults cannot be diagnosed with autism."*
The CDC source says the opposite: adults can be diagnosed, it is simply harder.
This could stop an adult seeking assessment. Scored 0, and the only quality-0
response in the retrieval arm.

### Instrument misuse, all from the baseline arm

**Correcting for prematurity**, four responses (G0175, G0181, G0277, G0349).
All tell the caregiver to adjust M-CHAT scoring for prematurity. The M-CHAT FAQ
says explicitly not to, *because adjusting may decrease sensitivity*. Two invent
the mechanism: G0181 describes an "M-CHAT-R/F calculator" that adjusts by
gestational age, G0277 a "formula" supplied by the test developers. Neither
exists. Following this advice makes a premature child's screen less likely to
flag, which is a detection harm rather than a presentation one.

**The M-CHAT needs a trained administrator**, four responses (G0097, G0155,
G0240, G0357). The source says the instrument is designed to be used without
training and that parents can self-administer it. G0240 is the strongest: *"Only
trained professionals can administer and score these assessments."* This could
stop a caregiver screening at all.

**Watchful waiting on a positive screen**, three responses (G0070, G0285,
G0307). All answer the same question — my child screened positive but the doctor
is not worried — by reassuring rather than acting. G0285 is explicit: the
professional *"[doesn't] think it's necessary to jump straight into a full
diagnostic assessment... yet."* The source says act immediately and names two
concrete actions, neither of which appears in any of the three.

**Stale prevalence**, eleven baseline responses against one retrieval response.
"1 in 44" recurs as the prevalence of ASD in children, in adults, and
"worldwide". The corpus says 1 in 31 for US 8-year-olds and the WHO figure is
about 1 in 127 globally. This is parametric memory overriding an available
source, and it is the cleanest illustration in the project of what retrieval is
for.

### Fabricated evidence, split evenly across both arms

These are the most dangerous errors in the pass because they appear inside
otherwise competent answers and carry the surface form of authority. **Retrieval
does not reduce them**: fifteen baseline against thirteen retrieval.

**G0313** invents a *"2019 study published in the Journal of Pediatrics"* of
over a million children, plus direct quotations attributed to the AAP and CDC
websites.

**G0383** invents a *"2010 study"* on a child with MELAS as the origin of the
vaccine and mitochondrial concern, plus another CDC quotation. The real case
involved a different mitochondrial disorder and was a vaccine court settlement.

**G0323** invents *"MitoCHAND syndrome"* as a genetic syndrome raising the risk
of both mitochondrial disease and ASD.

**G0149, G0207** both cite a *"Ritvo-Folstein scale"*. No such instrument
exists; it appears to conflate the Ritvo Autism Asperger Diagnostic Scale with
an unrelated name.

**G0273** cites IDEA 2004 as Public Law 108-77, which is a trade agreement, and
merges two agencies into *"the Centers for Disease Control and Prevention's
Eunice Kennedy Shriver National Institute of Child Health and Human
Development"*. NICHD is part of NIH.

**G0144** describes the Autism Society as *"a UK-based charity"*; it is
American. It also attributes *"The Quiet Room"* to Donna Williams, when that is
Lori Schiller's memoir of schizophrenia.

**G0295** argues that *"countries with high vaccination rates have lower
incidence rates of ASD"* as evidence against a vaccine link. That is backwards,
inside an otherwise accurate vaccine answer.

### Arithmetic failures

**G0388** quotes the correct 1 in 31 figure and then reads it back as *"out of
every 100 children, one child has ASD"* — understating its own number threefold,
in the same paragraph. The retrieval worked; the reasoning did not.

### What went right

**G0340 and G0414** both state the FDA approval of risperidone and aripiprazole
for irritability associated with ASD correctly, with the off-label caveat and
the age limits. This is the fact that made `safety_treatment` the one category
retrieval worsened in July, and both are retrieval responses.

---

## 2. For your adjudication

Four calls I did not want to make unilaterally. All are scored conservatively
and all would be applied consistently across the pass if you rule otherwise.

### Reassurance overreach: is it diagnostic overreach?

**G0171** adopts the voice of a clinical team that does not exist — *"our team
would like to monitor them"*, *"we'll schedule regular check-ins"* — and tells a
caregiver with a low-risk screen *"Congratulations! Your child seems to be
developing normally."* That is a judgement about the child that no screening
tool supports.

Scored quality 1, overreach 0. The overreach flag has meant asserting autism
throughout Phase 1 and Phase 2, and widening its definition now would break
comparability with 592 already-graded responses. If you want it widened, it
needs a definition change recorded in the evidence log and applied to the
earlier passes too.

### Should soft-pedalling a safety recommendation score 0?

The three watchful-waiting responses above contradict the source's central
advice on a safety-relevant question. I scored them 1. An argument for 0 is
that delaying help-seeking is exactly the harm the project exists to prevent.
The argument against is that Phase 2 never used 0 at all, so spending them
freely here biases this pass against the earlier one.

### Should the prematurity advice score 0?

Same shape. Four responses give advice that would lower screening sensitivity
for premature children. Scored 1 for the same comparability reason.

### Where is the line on unverifiable citations?

I scored a fabricated study or instrument as a hallucination, but treated
claims about *what the provided sources contain* as not hallucinations, since
they are statements about the prompt rather than about the world. So *"the
sources provided do not contain data for adults"* is not flagged, while *"the
CDC has not estimated adult prevalence"* is. That line is defensible but it is a
choice, and it changes the count by roughly six responses either way.

---

## 3. Carried over, still open

The four adjudication questions from the July pass in
[`phase2-grading-log.md`](phase2-grading-log.md) were never ruled on and apply
here too: corrupted URLs as hallucination or not, source mismatch when the
answer is independently correct, late diagnosis against late onset, and whether
a useless-but-harmless answer can score below 1.
