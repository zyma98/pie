"""
Parallel generation example using inferlib (Python).

Creates a shared system prompt context, then forks it into two independent
contexts that generate responses concurrently.  Both generations share the
KV cache from the common prefix.
"""

from inference_bindings import (
    Context,
    GenerateFuture,
    Model,
    SamplerConfig_Greedy,
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


def main() -> None:
    args = get_arguments()
    max_tokens = int(args.get("max_tokens", "128"))

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    common = Context(model)

    common.fill_system("You are a helpful, respectful and honest assistant.")
    poll_flush(common)

    stop_config = StopConfig(max_tokens=max_tokens, eos_sequences=eos_tokens)

    ctx1 = common.fork()
    ctx1.fill_user("Explain Pulmonary Embolism")
    future1 = ctx1.generate_async(SamplerConfig_Greedy(), stop_config)

    ctx2 = common.fork()
    ctx2.fill_user("Explain the Espresso making process ELI5.")
    future2 = ctx2.generate_async(SamplerConfig_Greedy(), stop_config)

    results = join_generate_futures([future1, future2])

    print(f"Output 1: {results[0]!r}")
    print(f"Output 2: {results[1]!r}")

    set_return(f"Output 1: {results[0]}\nOutput 2: {results[1]}")


if __name__ == "__main__":
    main()
