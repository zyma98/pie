"""
Prefix tree example using inferlib (Python).

Creates a 1x2x2x2 = 8 prompt tree structure and generates text concurrently
from all 8 leaf nodes, sharing KV cache from common prefixes.

Tree structure:
                      [System Prompt]
                    /                 \\
     [Photosynthesis]                 [Cellular Respiration]
     /              \\                     /                \\
[ELI5]        [High School]   [Location in Cell]     [Main Products]
/    \\           /        \\         /         \\          /    \\
[Chef] [Sunlight] [Equation] [Algae] [Mitochondria] [P&A]  [ATP] [CO2]
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


def _flush_all(ctxs: list[Context]) -> None:
    """Flush multiple contexts, polling all concurrently."""
    futures = [ctx.flush_async() for ctx in ctxs]
    active = {i: f for i, f in enumerate(futures) if f is not None}
    while active:
        indices = list(active.keys())
        pollables = [active[i].pollable() for i in indices]
        ready = poll(pollables)
        del pollables
        for ready_idx in ready:
            i = indices[ready_idx]
            if active[i].is_ready():
                del active[i]


def main() -> None:
    args = get_arguments()
    max_tokens = int(args.get("num_tokens", "128"))

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    stop_config = StopConfig(max_tokens=max_tokens, eos_sequences=eos_tokens)

    ctx_root = Context(model)

    # Level 0: Root
    ctx_root.fill_system(
        "You are a helpful, friendly, and knowledgeable science tutor for students "
        "of all ages. Your goal is to explain complex biological concepts in a clear, "
        "accessible, and engaging manner, tailoring your language to the specified audience."
    )
    poll_flush(ctx_root)

    # Level 1: Two topic branches
    ctx_photo = ctx_root.fork()
    ctx_photo.fill_user_only(
        "I'm curious about the fundamental process of photosynthesis. "
        "Could you provide a detailed overview of how plants create their own food "
        "using sunlight, water, and carbon dioxide?"
    )

    ctx_resp = ctx_root.fork()
    ctx_resp.fill_user_only(
        "Now, could you explain the equally important process of cellular respiration? "
        "I'd like to understand how organisms, including plants and animals, break down "
        "glucose to release the energy needed for life."
    )

    _flush_all([ctx_photo, ctx_resp])

    # Level 2: Four sub-topic branches
    ctx_photo_eli5 = ctx_photo.fork()
    ctx_photo_eli5.fill_user_only(
        "That sounds complicated. Could you simplify it significantly for me? "
        "Please explain the core idea in a way that a curious 5-year-old child could "
        "easily grasp and remember. Use a simple analogy."
    )

    ctx_photo_hs = ctx_photo.fork()
    ctx_photo_hs.fill_user_only(
        "Thank you. Now, could you provide a more technical explanation suitable for a "
        "high school biology student? I'm familiar with basic cell biology and chemistry, "
        "so please include relevant terminology like chloroplasts, chlorophyll, and "
        "light-dependent reactions."
    )

    ctx_resp_loc = ctx_resp.fork()
    ctx_resp_loc.fill_user_only(
        "I'm interested in the specific location within the cell where this process "
        "occurs. Can you describe the organelles involved and why their specific "
        "structures are uniquely suited for this essential energy-releasing function?"
    )

    ctx_resp_prod = ctx_resp.fork()
    ctx_resp_prod.fill_user_only(
        "Focusing on the outputs of this metabolic reaction, what are the primary "
        "products that result from this process? Please list and briefly describe "
        "the significance of each of these molecules for the cell."
    )

    _flush_all([ctx_photo_eli5, ctx_photo_hs, ctx_resp_loc, ctx_resp_prod])

    # Level 3: Eight leaf prompts
    leaf_ctxs: list[Context] = []

    p1 = ctx_photo_eli5.fork()
    p1.fill_user(
        "To make it really fun, please begin your explanation with the exact phrase "
        "'Plants are like little chefs...' and continue that cooking analogy to "
        "describe how they make their sugary food."
    )
    leaf_ctxs.append(p1)

    p2 = ctx_photo_eli5.fork()
    p2.fill_user(
        "Let's zoom in on the energy source for this recipe. Can you specifically "
        "detail the crucial role that sunlight plays in this process?"
    )
    leaf_ctxs.append(p2)

    p3 = ctx_photo_hs.fork()
    p3.fill_user(
        "For a more precise, scientific understanding, please provide the balanced "
        "chemical equation for the overall photosynthetic reaction."
    )
    leaf_ctxs.append(p3)

    p4 = ctx_photo_hs.fork()
    p4.fill_user(
        "How does this process in terrestrial plants compare to what happens in "
        "aquatic organisms like algae or cyanobacteria?"
    )
    leaf_ctxs.append(p4)

    p5 = ctx_resp_loc.fork()
    p5.fill_user(
        "Please elaborate specifically on the role of the mitochondria. Describe "
        "its inner and outer membranes and the matrix."
    )
    leaf_ctxs.append(p5)

    p6 = ctx_resp_loc.fork()
    p6.fill_user(
        "Is this metabolic pathway entirely identical in both plant and animal "
        "cells? Please compare and contrast the process."
    )
    leaf_ctxs.append(p6)

    p7 = ctx_resp_prod.fork()
    p7.fill_user(
        "One of the key products is usable energy. Could you explain in detail "
        "the role of adenosine triphosphate (ATP) as the main energy currency?"
    )
    leaf_ctxs.append(p7)

    p8 = ctx_resp_prod.fork()
    p8.fill_user(
        "I understand that carbon dioxide is considered a waste product of this "
        "process. Can you elaborate on what exactly happens to this CO2?"
    )
    leaf_ctxs.append(p8)

    # Generate all 8 concurrently
    print(f"--- Starting concurrent generation for 8 prompts (max {max_tokens} tokens each) ---")

    futures = [
        ctx.generate_async(SamplerConfig_Greedy(), stop_config) for ctx in leaf_ctxs
    ]
    results = join_generate_futures(futures)

    print(f"\n--- All 8 generations completed ---\n")

    for i, output_text in enumerate(results):
        print(f"Prompt #{i + 1}:\n{output_text!r}\n")

    set_return(f"Generated {len(results)} responses.")


if __name__ == "__main__":
    main()
