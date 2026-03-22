"""
Tree-of-Thought example using inferlib (Python).

Performs a 3-level tree search (Propose, Execute, Reflect) where each level
spawns multiple branches.  All branches at the same level are explored
concurrently, leveraging KV cache sharing from common prefixes.
"""

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

PROPOSE_PROMPT_TEMPLATE = (
    "Please generate a high-level plan for solving the following question. "
    "First, just state the method you will use. Do not do the actual calculation. "
    "Keep your response concise and within 80 words. Question: "
)

EXECUTE_PROMPT = (
    "The plan looks good! Now, use real numbers and do the calculation. "
    "Please solve the question step-by-step according to the plan. "
    "Give me the final answer. Make your response short."
)

REFLECT_PROMPT = (
    "Okay. Now, evaluate your own solution and give it a score on a scale of 1 to 5. "
    "Please rigorously check the correctness of the calculations and the final answer."
)


def main() -> None:
    args = get_arguments()
    question = args.get("question", "Calculate (42 + 3) * 5 / 15.")
    num_branches = int(args.get("num_branches", "2"))
    max_tokens_per_step = int(args.get("max_tokens", "512"))

    total_leaves = num_branches ** 3
    print(
        f"--- Starting Tree of Thought "
        f"(Branches={num_branches}, Leaves={total_leaves}, "
        f"MaxTokens/Step={max_tokens_per_step}) ---"
    )

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    sampler = SamplerConfig_TopP((0.6, 0.95))
    stop_config = StopConfig(max_tokens=max_tokens_per_step, eos_sequences=eos_tokens)

    ctx_root = Context(model)
    ctx_root.fill_system(
        "You are a helpful, respectful, and honest assistant that excels at "
        "mathematical reasoning. Please follow the user's instructions precisely."
    )
    poll_flush(ctx_root)

    # Level 1: Propose plans (concurrent)
    propose_ctxs = []
    propose_futures = []
    for _ in range(num_branches):
        propose_ctx = ctx_root.fork()
        propose_ctx.fill_user(f"{PROPOSE_PROMPT_TEMPLATE}{question}")
        propose_futures.append(propose_ctx.generate_async(sampler, stop_config))
        propose_ctxs.append(propose_ctx)

    join_generate_futures(propose_futures)

    # Level 2: Execute plans (concurrent across all branches)
    for propose_ctx in propose_ctxs:
        propose_ctx.fill_user(EXECUTE_PROMPT)
    poll_flush_all = [None] * len(propose_ctxs)
    for i, pctx in enumerate(propose_ctxs):
        poll_flush(pctx)

    execute_ctxs = []
    execute_futures = []
    for propose_ctx in propose_ctxs:
        for _ in range(num_branches):
            execute_ctx = propose_ctx.fork()
            execute_futures.append(execute_ctx.generate_async(sampler, stop_config))
            execute_ctxs.append(execute_ctx)

    join_generate_futures(execute_futures)

    # Level 3: Reflect on solutions (concurrent across all branches)
    for execute_ctx in execute_ctxs:
        execute_ctx.fill_user(REFLECT_PROMPT)
        poll_flush(execute_ctx)

    reflect_ctxs = []
    reflect_futures = []
    for execute_ctx in execute_ctxs:
        for _ in range(num_branches):
            reflect_ctx = execute_ctx.fork()
            reflect_futures.append(reflect_ctx.generate_async(sampler, stop_config))
            reflect_ctxs.append(reflect_ctx)

    join_generate_futures(reflect_futures)

    print(f"\n--- All {len(reflect_ctxs)} leaf nodes generated ---\n")

    if reflect_ctxs:
        print(
            f"Sample Result (Leaf #{len(reflect_ctxs)}):\n"
            f"{reflect_ctxs[-1].get_text()}\n"
        )
    else:
        print("No results were generated.")

    set_return(f"Generated {len(reflect_ctxs)} leaf results.")


if __name__ == "__main__":
    main()
