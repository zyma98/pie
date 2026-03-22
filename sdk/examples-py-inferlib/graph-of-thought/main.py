"""
Graph-of-Thought example using inferlib (Python).

Generates multiple initial proposals concurrently, then progressively
aggregates them in pairs across multiple levels.  Uses streaming completion
so aggregation can begin as soon as pairs of proposals are ready.
"""

from typing import Generator

from inference_bindings import (
    Context,
    GenerateFuture,
    Model,
    SamplerConfig_TopP,
    StopConfig,
    poll,
    set_return,
)
from run_bindings import get_arguments


def poll_flush(ctx: Context) -> None:
    """Block until a flush_async completes."""
    future = ctx.flush_async()
    if future is not None:
        pollable = future.pollable()
        pollable.block()
        del pollable
        del future


def join_generate_futures(futures: list[GenerateFuture]) -> list[str]:
    """Poll multiple GenerateFutures concurrently, return results in order."""
    results: list[str | None] = [None] * len(futures)
    remaining = dict(enumerate(futures))
    while remaining:
        indices = list(remaining.keys())
        pollables = [remaining[i].pollable() for i in indices]
        ready = poll(pollables)
        del pollables
        for ready_idx in ready:
            i = indices[ready_idx]
            result = remaining[i].get()
            if result is not None:
                results[i] = result
                del remaining[i]
    return results  # type: ignore[return-value]


def as_completed(
    futures: list[GenerateFuture],
) -> Generator[tuple[int, str], None, None]:
    """Yield (index, result) as GenerateFutures complete."""
    remaining = dict(enumerate(futures))
    while remaining:
        indices = list(remaining.keys())
        pollables = [remaining[i].pollable() for i in indices]
        ready = poll(pollables)
        del pollables
        for ready_idx in ready:
            i = indices[ready_idx]
            result = remaining[i].get()
            if result is not None:
                yield (i, result)
                del remaining[i]

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."

PROPOSAL_PROMPT_TEMPLATE = (
    "Could you suggest a method or approach to solve the following question? "
    "Please provide a high-level plan without doing the actual calculation. "
    "Keep it concise, around 80 words. Question: {}"
)

AGGREGATE_PROMPT = (
    "Please compare the following solution with the one you just provided "
    "and aggregate their ideas into a single, improved solution:\n"
)


def main() -> None:
    args = get_arguments()
    question = args.get("question", "Calculate (42 + 3) * 5 / 15.")
    proposal_tokens_str = args.get(
        "proposal_tokens", "256,256,256,256,256,256,256,256"
    )
    proposal_tokens = [int(s.strip()) for s in proposal_tokens_str.split(",")]
    aggregation_tokens = int(args.get("aggregation_tokens", "256"))

    print(f'--- Starting hierarchical aggregation for question: "{question}" ---')
    print(f"Proposal tokens: {proposal_tokens}, Aggregation tokens: {aggregation_tokens}")

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    sampler = SamplerConfig_TopP((0.6, 0.95))
    ctx_root = Context(model)

    ctx_root.fill_system(SYSTEM_PROMPT)
    propose_prompt = PROPOSAL_PROMPT_TEMPLATE.format(question)
    ctx_root.fill_user(propose_prompt)
    poll_flush(ctx_root)

    # --- Stage 1: Generate initial proposals concurrently ---
    proposal_futures = []
    proposal_ctxs = []
    for max_tok in proposal_tokens:
        ctx = ctx_root.fork()
        stop = StopConfig(max_tokens=max_tok, eos_sequences=eos_tokens)
        proposal_futures.append(ctx.generate_async(sampler, stop))
        proposal_ctxs.append(ctx)

    # Collect proposals as they complete and pair them for aggregation
    pending_pair: tuple[str, Context] | None = None
    first_agg_futures = []
    first_agg_ctxs = []

    for idx, proposal_text in as_completed(proposal_futures):
        proposal_ctx = proposal_ctxs[idx]
        if pending_pair is None:
            pending_pair = (proposal_text, proposal_ctx)
        else:
            prev_text, _ = pending_pair
            pending_pair = None
            proposal_ctx.fill_user(f"{AGGREGATE_PROMPT}{prev_text}")
            agg_stop = StopConfig(max_tokens=aggregation_tokens, eos_sequences=eos_tokens)
            first_agg_futures.append(proposal_ctx.generate_async(sampler, agg_stop))
            first_agg_ctxs.append(proposal_ctx)

    # --- Stage 2: Second-level aggregation (pair aggregation results) ---
    second_agg_futures = []
    second_agg_ctxs = []
    pending_agg: tuple[str, Context] | None = None

    for idx, agg_text in as_completed(first_agg_futures):
        agg_ctx = first_agg_ctxs[idx]
        if pending_agg is None:
            pending_agg = (agg_text, agg_ctx)
        else:
            prev_agg_text, _ = pending_agg
            pending_agg = None
            agg_ctx.fill_user(f"{AGGREGATE_PROMPT}{prev_agg_text}")
            agg_stop = StopConfig(max_tokens=aggregation_tokens, eos_sequences=eos_tokens)
            second_agg_futures.append(agg_ctx.generate_async(sampler, agg_stop))
            second_agg_ctxs.append(agg_ctx)

    # --- Stage 3: Collect final results ---
    final_solutions = join_generate_futures(second_agg_futures) if second_agg_futures else []

    print(f"\n--- Aggregation complete ---\n")
    for i, solution in enumerate(final_solutions):
        print(f"Final aggregated solution #{i + 1}:\n{solution}\n")

    set_return("\n\n".join(final_solutions) if final_solutions else "No solutions.")


if __name__ == "__main__":
    main()
