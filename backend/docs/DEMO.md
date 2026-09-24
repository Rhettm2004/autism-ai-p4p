# Live demo runbook

How to show the screening assistant working, live, in a meeting: a caregiver
question typed into a terminal, the router's decision, the retrieved sources with
links, and Llama's answer streaming in.

The front end is `scripts/chat.py`; the logic behind it is `src/chat.py`, tested
by `tests/test_chat.py`.

## What the audience is looking at

A turn is built with the same functions, prompt text, context format and router
configuration as the evaluated routed arm (llama3-8b, few_shot, top_k 5).

Because answers are read in a chat window, the demo adds one unevaluated
instruction by default: the concise style in the `chat:` block of
`config/prompts.yaml`. It asks for a direct answer of one or two sentences, up to
four short bullets, about 120 words and never more than 180, while keeping any
refusal and next step. It is appended after the measured prompt, never inserted
into it. `/concise off`, or starting with `--measured`, removes it, and
`tests/test_chat.py` asserts the prompt is then byte-identical to the one
`run_benchmark` builds. Say which mode is on when you show an answer: the concise
replies are not the ones the report graded.

Every turn prints:

| Panel | Shows |
| --- | --- |
| ROUTER | the route; whether the rule layer or the classifier decided; the rule pattern that fired; the classifier's top three probabilities; the full guidance text the router added to the prompt; a warning if the question is verbatim router training data |
| RETRIEVED | the top-k passages in rank order, with score, passage id and source number |
| ANSWER | the model's answer as it is generated, then latency and token counts |
| SOURCES | one numbered entry per source document, with its URL and authority tier; tier 5 is flagged (D-125) |

Every turn is also appended to `logs/demo/chat_<timestamp>.jsonl` with the full
system prompt, sources and response, so anything said in the meeting can be
looked up afterwards. `logs/` is gitignored.

## Before the meeting

On the GPU node (setup in the README, sections 1 to 9):

1. Book the workstation for the meeting slot and connect over the UoA VPN.
2. Start tmux so a dropped connection does not kill the loaded model:
   `tmux new -s demo`.
3. `conda activate autism-ai`, `cd ~/autism-ai`, `git pull`, and check out the
   branch carrying the demo.
4. Confirm `.env` sets `HF_HOME`, `huggingface-cli whoami` succeeds, and Meta's
   licence for Llama 3.1 has been accepted.
5. Make sure `data/corpus/corpus.csv` exists. It is gitignored, so either build
   it on the node (`python scripts/build_corpus.py`) or copy it across with
   `scp`.
6. `python scripts/preflight.py`. It includes a check that a demo turn builds
   and streams.
7. About ten minutes before the meeting, start the demo:
   `python scripts/chat.py`. Startup indexes the corpus, fits the router, loads
   the model and runs one warm-up generation, printing how long each took.
8. Enlarge the terminal font. Run `/ex 1` once to confirm the answer streams.

On the laptop, as a backup if the node is unavailable:

```bash
python scripts/chat.py --no-llm
```

This shows the router panel, retrieved passages, sources and, with `/prompt`,
the exact prompt the model would be sent. It needs the corpus locally and no GPU.

## Running order

`/examples` lists the scripted questions in `config/demo.yaml`, each copied
verbatim from a tracked prompt file and annotated with what it demonstrates.
`/ex N` asks one. The file is ordered as a running order: ordinary turns first,
then the safety turns.

Useful moves during the demo:

- **Same question, router off.** `/router off`, repeat the question, `/router on`.
  Only the guidance block changes. This is the routed-versus-retrieval
  comparison, live.
- **Same question, retrieval off.** `/rag off` gives the few-shot baseline, with
  no sources, which is the Phase 1 condition.
- **Concise versus measured.** `/concise off`, repeat the question, `/concise on`.
  The first answer is the evaluated system; the second is the chat-app style.
  The word count under each answer shows the difference.
- **Show the prompt.** `/prompt` prints the full system prompt of the last turn
  in labelled parts, in the order the model reads them: the condition prompt
  (safety rules and few-shot examples), the grounding rules, the route guidance
  the router added, the retrieved sources, then any demo-only instructions. The
  parts are found in the prompt that was actually sent, so the label on the
  route guidance is exactly what routing contributed.
- **Inline citations.** `/cite on` numbers the sources in the prompt and asks the
  model to cite them as [1], [2]. The answer panel labels this as not the
  evaluated configuration. Citations are renumbered in the order the answer
  first uses them, so they always read 1, 2, 3, and the sources list is
  reordered to match, with sources the model was given but did not cite listed
  after. The model can still attribute a sentence to the wrong source; check
  against the sources list.

## What to say about its limits

Cite the decision, not a number from memory. The numbers live in
`docs/report/numbers/`.

- **Routing is live here, replayed in the evaluation (D-119).** The evaluated arms
  used out-of-fold route decisions because the benchmark is the router's training
  data. Scripted questions taken from that data are flagged on screen.
- **The router's real safety recall is an open question (D-127).** A diagnosis
  request can be routed elsewhere; the held-out adversarial example is there to
  show it. The refusal rules live in the system prompt beneath every route, so a
  misrouted turn still carries them (checked by preflight).
- **Some sources are commercial or blog content (D-125).** They are flagged in the
  sources panel.
- **Answers are sampled.** Temperature is 0.1, not zero, so the same question can
  be worded differently on a second ask. `--seed` fixes it.
- **The concise style is not evaluated.** It was written after grading, for
  presentation. Whether shorter answers keep the rubric scores is untested, so
  quality claims belong to `/concise off` answers only.
- **Single turn only.** Each question is answered on its own with no conversation
  memory, matching the evaluation.
- **This does not call the Autism AI classifier.** It is the conversational layer
  around it.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No CUDA GPU is visible` | Run on the GPU node, or use `--no-llm`. Check `CUDA_VISIBLE_DEVICES`. |
| `Hugging Face refused access` | Accept Meta's licence on the model page, then `huggingface-cli login`. |
| `Corpus not found` | `python scripts/build_corpus.py`, or `scp` the corpus across. |
| First answer is slow | Startup's warm-up should prevent it; ask `/ex 1` before the meeting. |
| Colours garbled | `--no-color`, or set `NO_COLOR=1`. |
| Something odd happens | Restart with `--verbose` to see pipeline and library logging. |
