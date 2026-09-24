# Autism AI LLM

COMPSYS/SOFTENG 700 Project 18. Rayaan Rajabally and Rhett Murdoch.
Supervisor: Dr. Seyed Reza Shahamiri. Co-supervisor: Rabia Rao.

A conversational LLM layer wrapping the existing Autism AI screening classifier,
evaluated across three phases: baseline LLMs, retrieval-augmented generation
(RAG), and an agentic intent router.

---

## Contents

- [Quick start](#quick-start)
- [Live demo](#live-demo)
- [Repository layout](#repository-layout)
- [Local setup](#local-setup)
- [GPU setup on DeepNet](#gpu-setup-on-deepnet) (sections 1 to 9, unchanged)
- [Running the pipeline](#running-the-pipeline)
- [Keeping jobs running, copying results](#13-keeping-jobs-running-after-disconnect)
- [The RAG corpus](#the-rag-corpus)
- [Full run procedure](#full-run-procedure)
- [Benchmark versions](#benchmark-versions)
- [Troubleshooting](#troubleshooting)
- [Current state of results](#current-state-of-results)

---

## Quick start

From this directory, the repository root:

```bash
python run_all.py                       # every table and figure, no GPU, about 50s
python run_all.py --full --rag          # adds the GPU generation runs
python run_all.py --full --rag --arms   # adds the routed and union-control arms
python run_all.py --full --adversarial  # the held-out safety set, 20 prompts
```

With no flags it re-derives every number and figure from committed data and runs
on any machine. Add `--verbose` to stream stage output when something fails.

**Two things `run_all.py` deliberately does not do.** It does not run
`evaluate_router.py` or `evaluate_retrieval.py`, because those append a labelled
row to an experiment registry and re-running them from a convenience wrapper
would fill the history with duplicates. Run those by hand with a new `--label`
when you have actually changed something. And it does not grade: blind grading
is a person reading batches, and `prepare_grading.py` refuses to overwrite a
blinding map that annotations were scored against.

Individual commands:

```bash
python scripts/run_phase1.py --describe-benchmark
python scripts/preflight.py
python scripts/run_phase1.py --all --condition zero_shot few_shot
python scripts/run_phase1.py --all --condition zero_shot few_shot --rag --top-k 5 --expand-neighbours
python scripts/build_corpus.py
```

---

## Live demo

Ask the assistant questions in a terminal and watch it work: router A's decision,
the retrieved passages, the answer streaming in, and the sources as numbered
links. The default is llama3-8b, few_shot, retrieval top_k 5, the configuration
of the evaluated routed arm, plus an unevaluated concise chat-app style.
`--measured` or `/concise off` drops that style and sends the evaluated prompt
exactly.

```bash
python scripts/chat.py            # on a CUDA machine (the GPU node)
python scripts/chat.py --no-llm   # laptop backup: router, sources and prompt, no model
```

In the chat: `/examples`, `/ex N`, `/prompt`, `/concise on|off`, `/cite on|off`
(experimental inline citations), `/router on|off`, `/rag on|off`, `/help`, `/quit`. Transcripts go to
`logs/demo/`. The meeting runbook, including what to say about the demo's limits,
is [docs/DEMO.md](docs/DEMO.md).

---

## Repository layout

```
autism-ai/                      this repository, and the research compendium
├── run_all.py                  one command for the whole pipeline
├── config/
│   ├── models.yaml             model registry and generation settings
│   └── prompts.yaml            system prompts per condition, and the frozen
│                               route guidance and follow-up obligations
├── data/
│   ├── benchmark/              52-prompt benchmark, archived versions, and the
│   │                           held-out adversarial safety set
│   ├── corpus/
│   │   ├── sources.yaml        RAG source urls. EDIT THIS TO ADD SOURCES.
│   │   └── corpus.csv          built by build_corpus.py
│   ├── route_maps/             which route each prompt is treated as
│   ├── router_eval/            the router experiment registry
│   ├── retrieval_eval/         the retrieval experiment registry
│   ├── generation_eval/        arm comparisons and conformance tables
│   ├── grading*/               blinded batches, annotations, scored output
│   └── results/                run files, one row per response
├── docs/
│   ├── DATA_GUIDE.md           what every dataset is and what it cannot show
│   ├── decisions.md            what was decided, what is still open
│   ├── evidence/               append-only findings logs, cited by id
│   ├── figures/                figures, filed by phase
│   └── report/numbers/         every headline number, regenerated not typed
├── scripts/
│   ├── run_phase1.py           main generation entry point
│   ├── preflight.py            checks that must pass before a GPU booking
│   ├── build_corpus.py         fetch and chunk source pages
│   ├── build_route_maps.py     leak-free route assignments
│   ├── prepare_grading.py      blinded grading batches
│   ├── score_rag_rubric.py     applies the rubric, arm-aware
│   ├── evaluate_retrieval.py   Recall@k, MRR, chunk ablation
│   ├── evaluate_router.py      repeated-CV router evaluation
│   ├── evaluate_generation.py  N-way arm comparison
│   ├── evaluate_followup.py    route-appropriate follow-up conformance
│   ├── evaluate_adversarial.py the held-out safety set
│   ├── export_report_numbers.py
│   ├── make_report_figures.py
│   └── sync_report_notes.py
├── src/
│   ├── model_runner.py         generation, with optional retrieval and routing
│   ├── retrieval.py            chunking, indexing, retrieval
│   ├── router.py               Phase 3 intent router
│   ├── router_eval.py          repeated-CV harness for the router
│   ├── rubric.py               the grading scales and their coherence rules
│   ├── generation_control.py   completion-scoped repetition control
│   ├── degeneracy.py           repetition detection
│   ├── metrics.py              BERTScore, ROUGE-L, hallucination rate
│   └── benchmark_schema.py     validation
└── tests/                      eight standalone suites, no pytest needed
```

Everything needed to reproduce the work is in this repository. A `scoring/`
directory and an older `run_all.py` exist one level above from the mid-year
layout; they are not tracked here and nothing current depends on them.

---

## Local setup

Analysis and evaluation run without a GPU:

```bash
pip install pandas numpy pyyaml matplotlib scikit-learn scipy

# corpus fetching
pip install requests beautifulsoup4 pypdf

# reading a .numbers benchmark
pip install numbers-parser
```

---

## GPU setup on DeepNet

Unchanged from the original working procedure.

## Prerequisites

- UoA student account with access to the 2DN DeepNet portal
- Registered and approved at [http://deepnet.its.auckland.ac.nz](http://deepnet.its.auckland.ac.nz)
- Active booking on a workstation (you cannot SSH without one)
- HuggingFace account with access to:
  - `mistralai/Mistral-7B-Instruct-v0.1` (public)
  - `meta-llama/Meta-Llama-3.1-8B-Instruct` (request access at huggingface.co)
- GitHub Personal Access Token (PAT)

---

## 1. Booking a Workstation

1. Connect to UoA network or VPN
2. Go to [http://deepnet.its.auckland.ac.nz](http://deepnet.its.auckland.ac.nz)
3. Log in and make a booking for your assigned workstation
4. Note your start/end time and which GPU you booked

> You cannot SSH in without an active booking. Connection will be refused outside your slot.

---

## 2. Connecting via SSH

You must be on the UoA network or UoA VPN before connecting.

### Workstation IP addresses


| Workstation | IP Address     |
| ----------- | -------------- |
| Perseus     | 10.104.144.188 |
| Apollo      | 10.104.144.198 |
| Artemis     | 10.104.144.191 |
| Hermes      | 10.104.144.83  |
| Zeus        | 10.104.144.189 |
| Hades       | 10.104.147.2   |


### Connect

```bash
ssh yourUPI@IP_ADDRESS
```

For example:

```bash
ssh abcd123@10.104.144.188
```

Use your UoA university password when prompted. Nothing will appear on screen as you type -- that is normal.

### Close the connection when done

```bash
exit
```

---

## 3. Install Miniconda

Your account has no Python installed by default. Set up Miniconda first.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts:

- Hold Enter to scroll through the license or press Q to skip
- Type `yes` to accept
- Press Enter to confirm the default install location
- Type `yes` to initialise conda

Reload your shell. Note: if you are on `sh` instead of `bash`, use:

```bash
exec bash
```

You should see `(base)` at the start of your prompt.

---

## 4. Create and Activate Environment

```bash
conda create -n autism-ai python=3.10
conda activate autism-ai
```

Your prompt should change to `(autism-ai)`.

---

## 5. Clone the Repo

GitHub no longer accepts passwords for Git operations. Use a Personal Access Token instead.

Generate one at: github.com > Settings > Developer Settings > Personal Access Tokens > Tokens (classic)

- Select `repo` scope
- Copy the token immediately

Then clone:

```bash
git clone https://YOUR_TOKEN@github.com/Radiant-code/autism-ai.git
cd autism-ai
```

---

## 6. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 7. Install PyTorch with CUDA Support

The default PyTorch may not match the workstation CUDA driver. Reinstall with the correct version:

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

Check GPU status anytime with:

```bash
nvidia-smi
```

---

## 8. Configure HuggingFace Cache

```bash
cp .env.example .env
nano .env
```

Set the cache path. `/scratch` requires admin setup on some machines, so use your home directory if you get a permission error:

```
HF_HOME=/home/your-upi/hf_cache
```

Create the directory:

```bash
mkdir -p ~/hf_cache
```

---

## 9. Authenticate with HuggingFace

Get your token at huggingface.co > Settings > Access Tokens. Read scope is sufficient.

```bash
export HF_TOKEN=your_token_here
huggingface-cli login
```

Verify it worked:

```bash
huggingface-cli whoami
```

> For Llama 3.1 you must also accept Meta's license at:
> [https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
> Approval is usually within a few minutes to an hour.

---

---

## Running the pipeline

### Baseline generation

```bash
python scripts/run_phase1.py --model mistral-7b --condition zero_shot
python scripts/run_phase1.py --all --condition zero_shot few_shot
```

### With retrieval

```bash
python scripts/run_phase1.py --all --condition zero_shot few_shot --rag --top-k 5 --expand-neighbours
```

`--rag` requires `data/corpus/corpus.csv`. RAG results are written with a `rag3`
tag in the filename so they cannot be confused with baseline results.

When no retriever is supplied the behaviour is the Phase 1 baseline, unchanged.
When one is supplied, the top-k passages are injected into the system prompt
after the safety instructions, and the retrieved passage ids, source names and
retrieval latency are recorded against every response so any grounded answer can
be traced back to the text the model was shown.

### Metrics

```bash
python scripts/run_phase1.py --metrics data/results/YOUR_FILE.csv
```

Adds `bertscore_f1` and `rouge_l` columns. Every `reference_answer` in the
benchmark is grounded in its cited source, so no manual step is needed first.

| Metric | Weak | Acceptable | Good | Strong |
|---|---|---|---|---|
| BERTScore F1 | < 0.82 | 0.82 to 0.87 | 0.87 to 0.91 | > 0.91 |
| ROUGE-L F1 | < 0.10 | 0.10 to 0.20 | 0.20 to 0.35 | > 0.35 |

Always check by category rather than the overall mean. Phase 1 found these
metrics do not correlate with clinical quality (r = 0.084, p = 0.27), so treat
them as descriptive rather than as a basis for model selection.

### Evaluation, no GPU required

Everything below runs on any machine.

```bash
python run_all.py                        # all of the re-derivations at once

python scripts/evaluate_router.py --list # router configs and recorded steps
python scripts/export_report_numbers.py --list
python scripts/make_report_figures.py --list
```

The two experiments that append to a labelled registry are run by hand, with a
new label, when something has actually changed:

```bash
python scripts/evaluate_retrieval.py --label my_change --compare-to baseline
python scripts/evaluate_router.py --config cascade --label my_change --compare-to baseline
```

The re-derivations, which `run_all.py` runs for you:

```bash
python scripts/evaluate_generation.py --runs results_<date>     --ladder baseline rag routed --label end_to_end --scored <scored.csv>
python scripts/evaluate_followup.py --scored <scored.csv> --label <name>
python scripts/evaluate_adversarial.py   # routes the held-out safety set
```

---

## Phase 3: evaluating the intent router

The router needs no GPU: the benchmark's category labels already give the correct
route for every prompt, so routing is evaluated directly against them.

```bash
python scripts/evaluate_router.py --list
python scripts/evaluate_router.py --config baseline --label baseline
```

### Why every number is a mean with an interval

There are 93 labelled prompts spread over seven routes. A single cross-validation
split of a set that small is dominated by which examples happen to land in which
fold, so one accuracy figure cannot tell an improvement apart from a different
shuffle. Every evaluation therefore runs the whole cross-validation many times
over different shuffles and reports the mean with the half-width of its 95%
confidence interval.

Repeat `r` uses seed `base_seed + r`. Two configurations run with the same
`--repeats` and `--base-seed` therefore see identical folds, which is what makes
the comparison between them **paired**: the change is measured against itself on
the same data rather than against a separately shuffled run. The script refuses
to compare two runs whose seeds do not line up.

### Recording a change

Every router revision is recorded under a label, so the progression is a table
rather than a set of numbers copied by hand:

```bash
python scripts/evaluate_router.py --config char_ngrams --label char_ngrams     --compare-to baseline --notes "Word plus character 3-5 grams"
```

That appends a row to `data/router_eval/experiments.csv`, writes the per-repeat
scores, per-route breakdown, pooled confusion matrix and per-prompt correctness
rates beside it, and regenerates `docs/PHASE3_ROUTER_BRIEF.md`. Labels are unique:
re-running one is refused unless `--force` is passed, so numbers that have already
been reported cannot be quietly replaced. Use `--dry-run` to try something without
recording it.

### The router's representation

The adopted router unions frozen MiniLM sentence embeddings with word TF-IDF.
`sentence-transformers` downloads the encoder (~80MB) on first use, into `HF_HOME`
if it is set. The encoder runs on CPU and does not compete with the generation
models for GPU memory, and embeddings are cached by text within a process, so a
ten-repeat cross-validation encodes the prompt set once.

Router evaluation therefore needs network access the first time it runs, or a
pre-populated cache. Everything else about the router runs offline.

### Adding a router configuration

Register it in `src/router_eval.py` next to the existing entries:

```python
register_config(
    "my_variant",
    "One line saying what is different about it.",
    lambda: IntentRouter(...),
)
```

It is then available as `--config my_variant`. Nothing else changes.

### Tests

Eight standalone suites, no pytest needed. Each runs on its own and prints a
tally:

```bash
python tests/test_router_eval.py           # the router harness
python tests/test_retrieval_expansion.py   # neighbour expansion
python tests/test_generation_control.py    # completion-scoped repetition control
python tests/test_route_prompts.py         # route guidance and its injection
python tests/test_scoring_arms.py          # arm-aware rubric scoring
python tests/test_safety_grading.py        # the reference-free path and safety scales
python tests/test_followup_conformance.py  # route-appropriate follow-up
python tests/test_generation_comparison.py # the N-way arm comparison
```

They check the properties the reported numbers depend on: reproducibility under
fixed seeds, that repeats genuinely reshuffle, that two configurations are scored
on identical folds, that unpaired or duplicate comparisons are refused, and that
every benchmark category still maps to a route.

Several are pins rather than unit tests. `test_scoring_arms.py` asserts the
published two-arm rubric result is unchanged; `test_generation_comparison.py`
asserts the published truncation-fix comparison is unchanged, against a frozen
copy in `tests/fixtures/`; `test_route_prompts.py` asserts no route guidance
block contains a phrase the rubric's own regexes match, which would turn a
measurement of the model into a measurement of `config/prompts.yaml`. Those
failing means a published number has moved.

---

## 13. Keeping Jobs Running After Disconnect

Use tmux so your job survives if the SSH session drops:

```bash
tmux new -s phase1
```

Inside tmux, set up and run your job:

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=your_token_here
conda activate autism-ai
cd ~/autism-ai
python scripts/run_phase1.py --all --condition zero_shot few_shot
```


| Action                      | Command                       |
| --------------------------- | ----------------------------- |
| Detach (job keeps running)  | `Ctrl+B` then `D`             |
| Reattach after reconnecting | `tmux attach -t phase1`       |
| List sessions               | `tmux ls`                     |
| Kill session                | `tmux kill-session -t phase1` |


---

## 14. Copying Results to Your Local Machine

Run these commands on your **local machine** (not inside the SSH session).

### Copy a single results file

```bash
scp rraj313@10.104.144.188:~/autism-ai/data/results/filename.csv .
```

### Copy the entire results folder

```bash
scp -r rraj313@10.104.144.188:~/autism-ai/data/results/ .
```

The `.` at the end copies to your current local folder. Replace `rraj313` with your UPI.

> On Windows, open a new Command Prompt window and run the same command. OpenSSH is already installed if you were able to SSH in.

---

---

## The RAG corpus

### Adding a source

Edit `data/corpus/sources.yaml`. That is the only file to change.

```yaml
  - id: qchat10_scoring
    name: Q-CHAT-10 Scoring and Documentation
    url: https://example.org/qchat10.pdf
    authority: 3
    enabled: true
```

`name` must match the `source_name` column in the benchmark for retrieval
evaluation to score against it. `authority` is 1 for primary clinical guidelines,
2 for official public health pages, 3 for instrument documentation.

For a file behind a login or blocking scripted access, put it in
`data/corpus/manual/` and use `local:` instead of `url:`.

### Building

```bash
python scripts/build_corpus.py --list      # inspect the manifest
python scripts/build_corpus.py             # fetch and chunk
python scripts/build_corpus.py --refresh   # ignore cache
python scripts/build_corpus.py --chunk-size 80 --overlap 20
```

Writes `data/corpus/corpus.csv`. Raw downloads cache in `data/corpus/raw/`, so
re-chunking does not re-fetch.

### Why the corpus is source pages, not reference answers

The corpus is built from full source pages rather than the benchmark's extracted
reference answers. Indexing the answers would leak: retrieval would return the
answer, generation would paraphrase it, and BERTScore and ROUGE-L would rise
sharply without the system becoming any more reliable.

### Highest-value gap

Phase 1 hallucination concentrates on screening instrument details at 43.8%:
wrong Q-CHAT and M-CHAT age ranges, wrong item counts, reversed scoring
thresholds. The 14 seeded sources are general public-health pages and do not
cover these. Instrument documentation is the priority addition. The commented
block at the bottom of `sources.yaml` lists the specific gaps and the Phase 1
error each would fix.

---

## Full run procedure

Steps 1 to 3 need no GPU.

**1. Add missing sources** to `sources.yaml`.

**2. Build the corpus.**

```bash
python scripts/build_corpus.py
```

**3. Check retrieval, and check the generation path, before spending GPU time.**

```bash
python scripts/evaluate_retrieval.py --label my_change --compare-to baseline
python scripts/preflight.py
```

Retrieval gates everything: if recall is poor, retrieval cannot fix the Phase 1
hallucinations and the RAG runs will show nothing. Fix the corpus first.

Pre-flight gates the booking. It loads a small model on CPU, builds the longest
prompt the pipeline can produce, resolves every route and checks each arm writes
a distinct filename. It has already caught a `NameError` that would have failed
on every row after the model loaded, and a route map that did not cover the
benchmark. All checks must pass.

**4. Re-run the baseline.**

```bash
python scripts/run_phase1.py --all --condition zero_shot few_shot
```

Do this **before** any RAG run. The Phase 1 numbers currently in the report were
generated without repetition control. A RAG system that has it enabled is not
comparable against them, and running RAG first confounds the retrieval effect
with the generation-config change permanently.

**5. Run with retrieval.**

```bash
python scripts/run_phase1.py --all --condition zero_shot few_shot --rag --top-k 5 --expand-neighbours
```

**6. Compute metrics on every results file.**

```bash
for f in data/results/phase1_*.csv; do
  python scripts/run_phase1.py --metrics "$f"
done
```

**7. Grade the new responses, blind.** This is the only manual step and the one
that matters most: the rubric is the primary measure and no automated metric
substitutes for it.

```bash
python scripts/prepare_grading.py --results-dir results_<date>     --grading-dir data/grading_<date> --seed <n>
```

That writes batches with model, condition and retrieval status withheld, and a
`blind_map.csv` that must not be opened until grading is finished. Record one
row per response in `data/grading_<date>/annotations/batch_NN.csv`, then
unblind:

```bash
python scripts/score_rag_rubric.py --grading-dir data/grading_<date> --prefix <name>
```

`validate_annotations` in `src/rubric.py` refuses a malformed grade rather than
letting it average into a rate, and `prepare_grading.py` refuses to overwrite an
existing blinding map, which would orphan every annotation scored against it.

**8. Regenerate everything.**

```bash
python run_all.py
```

---

## Benchmark versions

**Benchmark revisions reuse prompt ids for different questions.** Old P001 was a
Q-CHAT pointing question; current P001 is "Do vaccines cause ASD?". Scoring a
results file against the wrong version would join cleanly on id and produce
plausible but meaningless numbers.

The pipeline compares prompt text, not just ids, and refuses to proceed on a
mismatch. Archived versions are in `data/benchmark/archive/`:

| File | Prompts | Used by |
|---|---|---|
| `phase1_benchmark_v2_44prompt_reconstructed.csv` | 44 | May 2026 result runs |
| `phase1_baseline_benchmark_v1_40prompt.csv` | 40 | earlier draft |
| `phase1_rubric_benchmark_25prompt.csv` | 25 | rubric draft |

To score an older results file:

```bash
python scripts/run_phase1.py --metrics results.csv \
  --benchmark data/benchmark/archive/phase1_benchmark_v2_44prompt_reconstructed.csv
```

To convert a new `.numbers` benchmark:

```bash
python scripts/convert_numbers_benchmark.py path/to/new.numbers --force
```

---

## Troubleshooting

**`Corpus not found`** with `--rag`. Run `scripts/build_corpus.py` first.

**`Annotation table is missing columns` or `Quality must be 0-3`** from
`score_rag_rubric.py`. A grade is malformed. `validate_annotations` in
`src/rubric.py` refuses it rather than letting it average into a rate, and the
message names the offending `grade_id`s. This is intended behaviour.

**`... already exists. Grading it again would orphan the annotations`** from
`prepare_grading.py`. Use a new `--grading-dir` for a new pass. Regenerating a
map that annotations were scored against breaks the link between a grade and the
response it belongs to, and `--force` is almost never what you want.

**`the reference-free set validated without --no-reference`** or a
`BenchmarkValidationError` about `reference_answer`. The adversarial safety set
carries no reference answers, because a refusal has no single correct text to be
compared against. Add `--no-reference`, and give it its own `--output-dir`.

**`Prompt text mismatch on N rows`.** The results file and benchmark are
different versions. Point `--benchmark` at the archived version those results
were generated with.

**`load_in_4bit requested but CUDA is not available`.** No GPU visible.
`bitsandbytes` 4-bit requires CUDA, so this falls back to fp32 and needs roughly
28GB of RAM. Fix the GPU setup rather than continuing.

**A source fails to fetch.** Usually a blocked user agent. Download the page by
hand into `data/corpus/manual/` and switch that entry to `local:`.

**A stage is skipped by `run_all.py`.** Its inputs do not exist yet. Before the
first model run the rubric and figure stages have nothing to work on, which is
expected.

**Stale git locks.** Delete `.git/index.lock` and `.git/HEAD.lock`.

---

## Current state of results

Last updated 26 August 2026. `docs/DATA_GUIDE.md` is the authority on every
dataset; `docs/decisions.md` records what was decided and what is still open.

**Phase 1 is complete.** 176 responses graded by hand against the rubric.
Ungrounded models are safe at the diagnosis boundary and wrong about the
screening instruments.

**Phase 2 is complete and measured twice.** The 25 August re-run generated both
arms together on the 52-prompt benchmark and graded all 416 responses blind in
one pass. Retrieval cuts hallucination from 30.3% to 11.1% and raises mean
quality from 1.889 to 2.375, paired per prompt, significant in every run. That
is a clean within-pass comparison. Comparing across the July and August passes
is not, and is caveated wherever it appears (D-116, D-117).

**Retrieval also stopped the model aiming its follow-up advice.** The flat rate
is unchanged, but the gap between turns that owe a next step and turns that do
not falls from +24.5pp to −0.8pp (F-P2-015). Ten of the 52 prompts owe a next
step, so this rests on ten prompts and says so.

**Phase 3 has a router and a measurement harness; the end-to-end arm does not
exist yet.** The adopted router scores 86.9% under repeated cross-validation and
56.9% when fitted on non-benchmark prompts only, and the report must give both
and say which question each answers (D-118). Route-conditioned prompting is
built, frozen and pre-flight checked, but no routed response has been generated,
so **no claim that routing improves generation is yet supported by anything**.

**The held-out adversarial safety set has been routed but not run.** The adopted
router reaches the safety route on 13 of 20; the rule layer alone catches 1. The
seven it misses still build prompts carrying the refusal rules, which is checked
mechanically, but whether the model obeys them is what the run will measure
(F-E2E-002).

**Still outstanding.** No second grader and no inter-rater agreement figure
(D-105), so every rubric number is one annotator's judgement. The adversarial
set was written by the same author as the rule patterns, making it a worst case
against an informed adversary rather than an estimate of ordinary use (D-107).
