# References

Sources used or needed, what each is used for, and whether the full citation has
been verified. Anything marked **verify** has a plausible reference here but the
exact authors, year or edition have not been checked against the source; do not
put it in the report until it has been.

## Clinical criteria and guidelines

| Source | Used for | Status |
|---|---|---|
| ICD-11 Section 6A02, WHO | Autism definition and criteria | Cited in project brief. Verify edition and access date. |
| DSM-5, American Psychiatric Association | Diagnostic criteria for ASD | **verify** full citation |
| NICE CG128, Autism spectrum disorder in under 19s: recognition, referral and diagnosis | Referral pathway; RAG corpus source | In corpus (`sources.yaml`). Verify publication and update dates. |
| New Zealand Autism Spectrum Disorder Guideline, 3rd edition | NZ clinical guidance; RAG corpus source | In corpus. **verify** publisher and year |
| CDC, Autism prevalence (1 in 31) | Current prevalence; the figure the models got wrong | In corpus via `manual/cdc_prevalence.txt`. Verify ADDM survey year. |
| WHO, Autism fact sheet | General definitions | In corpus |
| NIMH, Autism spectrum disorder | General knowledge answers | In corpus |
| NICHD | Symptoms, causes, treatments, early intervention | In corpus, five pages |

## Screening instruments

| Source | Used for | Status |
|---|---|---|
| Q-CHAT-10, Allison et al. | The 10-item toddler instrument, its age range and cutoff | Paper is in the corpus manifest but **disabled** for retrieval (D-007). **verify** citation, commonly Allison, Auyeung & Baron-Cohen, 2012 |
| M-CHAT-R/F, Robins et al. | The 20-item instrument, scoring and follow-up | In corpus (FAQ, scoring, guidelines pages). **verify**, commonly Robins et al., 2014 |
| Autism AI, Shahamiri and Thabtah | The screening classifier this project wraps | **verify** the correct paper for the deployed model at autism.rezanet.com |

## Methods

| Source | Used for | Status |
|---|---|---|
| BERTScore, Zhang et al. | Semantic similarity metric, reported as descriptive only | **verify**, commonly Zhang, Kishore, Wu, Weinberger & Artzi, ICLR 2020 |
| ROUGE, Lin | Lexical overlap metric | **verify**, commonly Lin, 2004 |
| Retrieval-augmented generation, Lewis et al. | The Phase 2 approach | **verify**, commonly Lewis et al., NeurIPS 2020 |
| Mistral 7B | One of the two evaluated models | **verify**, Jiang et al., 2023 |
| Llama 3 | The other evaluated model | **verify**, Meta AI, 2024 |
| TF-IDF / lexical retrieval baseline | The retriever | Needs a citation; consider Robertson and Spärck Jones for the underlying weighting |

## Corpora

| Source | Used for | Status |
|---|---|---|
| TalkBank ASDbank | Language corpus from individuals with ASD | Named in the project brief; **not currently used** by any phase. Either use it or drop it from the report. |

## Still needed

The literature review threads in [`report/outline.md`](report/outline.md) need
sources that the project does not yet cite:

- LLMs in clinical communication and patient-facing health information, including
  the evidence on hallucination in medical question answering.
- Agentic and multi-agent LLM architectures, and specifically the argument that
  safety-critical domains favour hybrid symbolic-neural designs. The mid-year
  report asserts this without a citation, and the Phase 3 design rests on it.
- Intent classification and routing for dialogue systems, to place the router in
  a literature rather than presenting it as an ad hoc component.
- Over-refusal as a documented failure mode of safety-tuned models, which is
  exactly what Llama3-8B did in Phase 1.

The third and fourth are the most important, because they are the two places
where our own findings would otherwise look like isolated observations rather
than instances of a known phenomenon.
