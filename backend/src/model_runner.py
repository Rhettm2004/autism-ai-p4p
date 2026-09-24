"""Model loading, inference, and benchmark orchestration."""

import time
from datetime import datetime
from pathlib import Path

import yaml

from src.benchmark_schema import validate_benchmark_csv
from src.degeneracy import degeneracy_report
from src.utils import get_logger, load_benchmark, save_results

logger = get_logger(__name__)

# torch and transformers are imported lazily inside the functions that need them.
# This keeps benchmark validation, schema inspection, and config loading usable on
# machines without the GPU stack installed (e.g. a laptop preparing a run).

_MODELS_YAML = Path(__file__).parent.parent / "config" / "models.yaml"
_PROMPTS_YAML = Path(__file__).parent.parent / "config" / "prompts.yaml"


def load_model_config(model_id: str) -> dict:
    """
    Return the config dict for model_id from models.yaml.

    Raises ValueError listing available ids if model_id is not found.
    """
    with open(_MODELS_YAML, encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    models = registry.get("models", {})
    if model_id not in models:
        available = ", ".join(models.keys())
        raise ValueError(
            f"Model '{model_id}' not found in models.yaml. "
            f"Available models: {available}"
        )
    return models[model_id]


def load_model_and_tokenizer(config: dict) -> tuple:
    """
    Load a HuggingFace model and tokenizer from a config dict.

    Uses 4-bit quantisation when config['load_in_4bit'] is True.
    Always uses device_map='auto'.
    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_id = config["hf_model_id"]
    logger.info("Loading tokenizer for %s", hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    cuda_available = torch.cuda.is_available()
    use_4bit = config.get("load_in_4bit", False) and cuda_available

    if config.get("load_in_4bit", False) and not cuda_available:
        logger.warning("load_in_4bit requested but CUDA is not available — loading in fp32 on CPU")

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    dtype = torch.float16 if cuda_available else torch.float32
    logger.info("Loading model %s (4-bit=%s, device=%s)", hf_id, use_4bit, "cuda" if cuda_available else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=dtype,
    )
    logger.info("Model loaded: %s", hf_id)
    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    """Delete model and tokenizer references and free CUDA memory."""
    import torch

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded and CUDA cache cleared")


def generate_response(
    prompt: str,
    system_prompt: str,
    tokenizer,
    model,
    config: dict,
    seed: int | None = None,
    streamer=None,
) -> tuple[str, float, int, int]:
    """
    Generate a single response.

    Returns (response_text, latency_ms, prompt_token_count, completion_token_count).
    Falls back to manual [INST] formatting if apply_chat_template raises.

    streamer is an optional transformers streamer, used only by the interactive
    demo (scripts/chat.py) so tokens appear as they are generated. It observes
    generation without altering it, and batch runs never pass one.
    """
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = (
            f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_token_count = inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": config.get("max_new_tokens", 512),
        "temperature": config.get("temperature", 0.1),
        "do_sample": config.get("do_sample", True),
        "pad_token_id": tokenizer.eos_token_id,
    }
    # Repetition control. Only passed when configured, so omitting them from
    # models.yaml reproduces the original Phase 1 generation behaviour exactly.
    #
    # Scope decides who applies them. Under "sequence" they go to generate() and
    # the stock processors read the whole sequence, prompt included, which is
    # what produced the truncation in the Phase 2 RAG runs (F-P2-010). Under
    # "completion" the same constraints are applied to the generated text only,
    # so quoting a retrieved source is no longer treated as repetition.
    scope = config.get("repetition_scope", "completion")
    if scope == "sequence":
        if config.get("repetition_penalty") is not None:
            gen_kwargs["repetition_penalty"] = config["repetition_penalty"]
        if config.get("no_repeat_ngram_size") is not None:
            gen_kwargs["no_repeat_ngram_size"] = config["no_repeat_ngram_size"]
    else:
        # Imported here, not at module scope: generation_control pulls in
        # transformers, and this module stays importable without the GPU stack.
        from src.generation_control import build_logits_processors

        processors = build_logits_processors(config, prompt_token_count)
        if len(processors):
            gen_kwargs["logits_processor"] = processors

    # Sampling is on (do_sample true, temperature 0.1), so two runs of the same
    # prompt draw different tokens. Between the July and August passes the
    # baseline moved 7.7pp on identical settings and the cause was never
    # established (D-117). That drift is larger than the effects the arms are
    # meant to measure, so when a seed is supplied the draw is fixed. Callers
    # derive it from the prompt's position rather than the arm, so a prompt
    # starts from the same RNG state in every arm and the comparison is paired
    # at the sampling level rather than merely averaged over.
    if seed is not None:
        torch.manual_seed(seed)

    if streamer is not None:
        gen_kwargs["streamer"] = streamer

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    generated_ids = output_ids[0][prompt_token_count:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    completion_token_count = len(generated_ids)

    return response_text, latency_ms, prompt_token_count, completion_token_count


def load_route_guidance(route: str) -> str:
    """
    Return the guidance block for a route, from config/prompts.yaml.

    Raises rather than falling back to no guidance. A typo in a route name would
    otherwise silently degrade a routed arm into a plain retrieval arm on the
    prompts it affects, and the run would look like it had worked.
    """
    with open(_PROMPTS_YAML, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    routes = config.get("routes", {})
    if route not in routes:
        available = sorted(k for k in routes if k != "_meta")
        raise ValueError(
            f"Route '{route}' has no guidance in prompts.yaml. "
            f"Available: {', '.join(available)}"
        )
    return routes[route]["guidance"].strip()


def build_grounded_system_prompt(system_prompt: str, context: str,
                                 route_guidance: str | None = None) -> str:
    """
    Combine the safety instructions with retrieved context.

    Context is placed after the instructions and framed explicitly, so the
    safety constraints are never displaced by retrieved text, and the model is
    told to defer to the sources rather than to its own recollection. Phase 1
    showed the failures are confident wrong facts, so the instruction to say
    when the context does not cover something matters as much as the context.
    """
    return (
        f"{system_prompt}\n\n"
        "Use the following source extracts to answer. These are authoritative "
        "and take precedence over anything you recall. If they do not contain "
        "the answer, say so rather than guessing. Do not invent details about "
        "screening instruments, age ranges, or scoring thresholds.\n\n"
        # Grounded responses in Phase 2 quoted the source and stopped: advice to
        # see a professional fell from 27.9% to 18.8% and the share answering the
        # question at all fell from 95.7% to 85.1% (F-P2-009). The sources
        # instruction was displacing the caregiver-facing instructions above it,
        # so those are restated after it rather than before.
        "Grounding your answer does not replace the instructions above. Answer "
        "the question that was asked rather than only quoting, keep the plain "
        "language a caregiver needs, and still say what the next step is and "
        "when to speak to a professional.\n\n"
        # Route guidance sits between the restated caregiver rules and the
        # sources: adjacent to the behaviour it specialises, and not competing
        # with the safety rules at the top. Omitted, the string is byte-identical
        # to the unrouted pipeline, which is what keeps earlier runs reproducible.
        + (f"{route_guidance}\n\n" if route_guidance else "")
        + f"--- SOURCES ---\n{context}\n--- END SOURCES ---"
    )


def run_benchmark(
    model_id: str,
    condition: str,
    benchmark_csv: str,
    output_dir: str,
    retriever=None,
    expand_neighbours: bool = False,
    top_k: int = 3,
    seed: int | None = None,
    arm: str = "",
    routes: dict | None = None,
    route_source: str = "",
    allow_missing_reference: bool = False,
) -> str:
    """
    Run a full benchmark evaluation for one model and one condition.

    Loads the model, iterates every row in benchmark_csv, generates a response,
    unloads the model, saves results, and returns the output file path.

    When retriever is None the behaviour is the Phase 1 baseline, unchanged.
    When a retriever is supplied, the top_k passages for each prompt are injected
    into the system prompt before generation, which is the Phase 2 condition. The
    retrieved passage ids and sources are recorded per response so any answer can
    be traced back to what the model was actually shown.
    """
    with open(_PROMPTS_YAML, encoding="utf-8") as f:
        prompts_cfg = yaml.safe_load(f)

    available_conditions = list(prompts_cfg.get("conditions", {}).keys())
    if condition not in prompts_cfg.get("conditions", {}):
        raise ValueError(
            f"Condition '{condition}' not found in prompts.yaml. "
            f"Available conditions: {', '.join(available_conditions)}"
        )
    system_prompt = prompts_cfg["conditions"][condition]["system_prompt"]

    # Validate the benchmark and the model config BEFORE loading weights, so a
    # malformed benchmark fails in seconds instead of after a multi-GB model load.
    benchmark_df = validate_benchmark_csv(
        benchmark_csv, allow_missing_reference=allow_missing_reference)
    model_config = load_model_config(model_id)

    logger.info(
        "Benchmark %s: %d prompts across %d categories",
        benchmark_csv,
        len(benchmark_df),
        benchmark_df["category"].nunique(),
    )

    model, tokenizer = load_model_and_tokenizer(model_config)

    rows = load_benchmark(benchmark_csv)
    total = len(rows)
    results: list[dict] = []

    for index, row in enumerate(rows, start=1):
        prompt_id = row.get("prompt_id", "unknown")
        category = row.get("category", "unknown")
        prompt_text = row.get("prompt", "")
        logger.info(
            "[%d/%d] Running prompt_id=%s category=%s", index, total, prompt_id, category
        )

        # Phase 3: the route decides what this kind of turn owes the caregiver.
        # Resolved before the retrieval branch so a routed arm without retrieval
        # stays possible; injected inside it, because the guidance is part of the
        # grounded prompt. A prompt id missing from a supplied map is an error,
        # not a silent fallback to plain retrieval.
        route = ""
        route_guidance = None
        if routes is not None:
            if prompt_id not in routes:
                raise ValueError(
                    f"No route for prompt_id={prompt_id} in the supplied map. "
                    f"The map must cover every benchmark prompt."
                )
            route = routes[prompt_id]
            # An empty route means the router did not reach a decision. The LLM
            # router can emit output that names no label, and the honest
            # response to that is to generate unrouted rather than to fall back
            # on a default route: a default would silently attribute one route's
            # guidance to a turn nothing classified, and the arm would be
            # measuring that default rather than the router.
            if route:
                route_guidance = load_route_guidance(route)

        # Phase 2: ground the system prompt in retrieved sources.
        turn_system_prompt = system_prompt
        retrieved_ids, retrieved_sources, retrieval_ms = "", "", 0.0
        if retriever is not None:
            t_r = time.perf_counter()
            try:
                hits = retriever.retrieve(prompt_text, k=top_k,
                                          expand_neighbours=expand_neighbours)
                retrieval_ms = (time.perf_counter() - t_r) * 1000
                if hits:
                    context = "\n\n".join(
                        f"[Source: {p.source_name}]\n{p.text}" for p, _ in hits
                    )
                    turn_system_prompt = build_grounded_system_prompt(
                        system_prompt, context, route_guidance)
                    retrieved_ids = "|".join(p.passage_id for p, _ in hits)
                    retrieved_sources = "|".join(
                        dict.fromkeys(p.source_name for p, _ in hits)
                    )
                else:
                    logger.warning("No passages retrieved for prompt_id=%s", prompt_id)
            except Exception as exc:
                logger.error("Retrieval failed for prompt_id=%s: %s", prompt_id, exc)

        try:
            # Derived from the prompt's position, never from the arm, so the
            # same prompt starts from the same RNG state in every arm.
            turn_seed = None if seed is None else seed + index
            response, latency_ms, prompt_tokens, completion_tokens = generate_response(
                prompt=prompt_text,
                system_prompt=turn_system_prompt,
                tokenizer=tokenizer,
                model=model,
                config=model_config,
                seed=turn_seed,
            )
            error = ""
        except Exception as exc:
            logger.error("Error on prompt_id=%s: %s", prompt_id, exc)
            response, latency_ms, prompt_tokens, completion_tokens = "", 0.0, 0, 0
            error = str(exc)

        results.append(
            {
                "prompt_id": prompt_id,
                "category": category,
                "prompt": prompt_text,
                "model_id": model_id,
                "condition": condition,
                "response": response,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                # Source provenance is carried through from the benchmark so a
                # results file is self-describing without re-joining.
                "source_name": row.get("source_name", ""),
                "answer_origin": row.get("answer_origin", ""),
                # Phase 2 provenance. Empty for baseline runs, which is how the
                # two conditions are told apart in the results file.
                "rag_enabled": int(retriever is not None),
                # Recorded per row so a results file says how it was produced.
                # Without it the new runs are indistinguishable from the Phase 2
                # ones by content alone, and the grading pass reads these files
                # months later.
                "retrieval_expanded": int(retriever is not None and expand_neighbours),
                "repetition_scope": model_config.get("repetition_scope", "completion"),
                # Arm and seed identify which experiment a row belongs to. All
                # retrieval arms share rag_enabled=1, so nothing else separates
                # them once the files are pooled for scoring.
                "arm": arm,
                "seed": "" if seed is None else seed + index,
                "route_assigned": route,
                "route_source": route_source,
                "retrieved_passage_ids": retrieved_ids,
                "retrieved_sources": retrieved_sources,
                "retrieval_ms": round(retrieval_ms, 2),
                **degeneracy_report(response),
                "timestamp": datetime.utcnow().isoformat(),
                "error": error,
            }
        )
        if degeneracy_report(response)["degenerate"]:
            logger.warning("Degenerate output detected for prompt_id=%s", prompt_id)

    unload_model(model, tokenizer)
    # Every retrieval arm carries rag_enabled=1, so the arm has to reach the
    # filename: prepare_grading.py's only human-readable provenance is source_run,
    # and three arms sharing a stem cannot be told apart once graded.
    tag = "_".join(part for part in
                   (f"rag{top_k}" if retriever is not None else "", arm) if part)
    output_path = save_results(results, model_id, condition, output_dir, tag=tag)
    logger.info("Results saved to %s", output_path)
    return output_path
