"""
Skeleton-of-Thought example using inferlib (Python).

First generates a high-level plan (skeleton) with key points, then elaborates
on each point concurrently.  This reduces latency by parallelising the
detailed generation phase.
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


def poll_generate(future: GenerateFuture) -> str:
    """Block until a GenerateFuture produces a result."""
    while True:
        pollable = future.pollable()
        pollable.block()
        del pollable
        result = future.get()
        if result is not None:
            return result


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


def main() -> None:
    args = get_arguments()
    question = args.get(
        "question", "What are the defining characteristics of Rome?"
    )
    num_points = int(args.get("num_points", "3"))
    plan_max_tokens = int(args.get("plan_tokens", "256"))
    elab_max_tokens = int(args.get("elab_tokens", "256"))

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    ctx = Context(model)

    ctx.fill_system("You are a helpful, respectful and honest assistant.")
    poll_flush(ctx)

    sampler = SamplerConfig_TopP((0.6, 0.95))

    # --- Phase 1: Generate plan ---
    plan_ctx = ctx.fork()
    plan_prompt = (
        f"Generate up to {num_points} key points that outline the answer to "
        f"the following question: {question}. "
        "Each point must be enclosed between the <point> and </point> tags."
    )
    plan_ctx.fill_user(plan_prompt)

    plan_stop = StopConfig(max_tokens=plan_max_tokens, eos_sequences=eos_tokens)
    plan_future = plan_ctx.generate_async(sampler, plan_stop)
    plan_output = poll_generate(plan_future)

    # Parse <point>...</point> tags
    points: list[str] = []
    for segment in plan_output.split("<point>")[1:]:
        end = segment.find("</point>")
        if end != -1:
            text = segment[:end].strip()
            if text:
                points.append(text)

    if not points:
        print("No points were generated or elaborated upon.")
        set_return("No points generated.")
        return

    # --- Phase 2: Elaborate on each point concurrently ---
    elab_stop = StopConfig(max_tokens=elab_max_tokens, eos_sequences=eos_tokens)
    futures = []
    for point in points:
        elab_ctx = ctx.fork()
        elab_ctx.fill_user(
            f"Elaborate on the following point: {point}. "
            "Your response should be complete and only concerned with this point."
        )
        futures.append(elab_ctx.generate_async(sampler, elab_stop))

    elaborations = join_generate_futures(futures)

    print(f"\n--- {len(elaborations)} elaborations completed ---\n")
    for i, elab in enumerate(elaborations):
        print(f"Elaboration {i + 1}:\n{elab}\n")

    set_return("\n\n".join(elaborations))


if __name__ == "__main__":
    main()
