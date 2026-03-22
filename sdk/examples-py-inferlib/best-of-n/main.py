"""
Best-of-N generation with diversity ranking using inferlib (Python).

Forks a context N times to generate N candidate responses in parallel,
then uses string similarity (difflib.SequenceMatcher) to compute pairwise
similarity and select the most central (consensus) answer.
"""

from difflib import SequenceMatcher

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

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step by step. "
    "Show your reasoning, then give your final answer on the last line "
    "in the format: Final Answer: <answer>"
)


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


def extract_final_answer(response: str) -> str:
    """Extract the text after 'Final Answer:' from a response."""
    idx = response.rfind("Final Answer:")
    if idx >= 0:
        return response[idx + len("Final Answer:"):].strip()
    return response.strip()


def truncate(s: str, max_len: int) -> str:
    s = s.replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def normalized_similarity(a: str, b: str) -> float:
    """Compute normalized string similarity using SequenceMatcher (0..1)."""
    return SequenceMatcher(None, a, b).ratio()


def main() -> None:
    args = get_arguments()
    question = args.get("question", "What is 17 * 24 + 13?")
    num_candidates = int(args.get("num_candidates", "5"))
    max_tokens = int(args.get("max_tokens", "1024"))

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()

    base_ctx = Context(model)
    base_ctx.fill_system(SYSTEM_PROMPT)
    base_ctx.fill_user(str(question))
    poll_flush(base_ctx)

    stop_config = StopConfig(max_tokens=max_tokens, eos_sequences=eos_tokens)

    # --- Stage 1: Generate N candidates in parallel ---
    print(f"--- Generating {num_candidates} candidates in parallel ---")

    futures: list[GenerateFuture] = []
    for _ in range(num_candidates):
        ctx = base_ctx.fork()
        future = ctx.generate_async(SamplerConfig_TopP((0.6, 0.95)), stop_config)
        futures.append(future)

    candidates = join_generate_futures(futures)
    print(f"Generated {len(candidates)} candidates\n")

    # --- Stage 2: Extract final answers ---
    answers = [extract_final_answer(c) for c in candidates]

    print("--- Extracted Answers ---\n")
    for i, answer in enumerate(answers):
        print(f"  Candidate {i + 1}: \"{truncate(answer, 80)}\"")
    print()

    # --- Stage 3: Compute pairwise similarity on extracted answers ---
    print("--- Computing pairwise similarity ---")

    n = len(candidates)
    similarity_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        similarity_matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = normalized_similarity(answers[i], answers[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim

    # --- Stage 4: Rank by centrality ---
    centrality_scores: list[float] = []
    for i in range(n):
        if n == 1:
            centrality_scores.append(1.0)
        else:
            total = sum(similarity_matrix[i][j] for j in range(n) if j != i)
            centrality_scores.append(total / (n - 1))

    best_idx = max(range(n), key=lambda i: centrality_scores[i])

    # --- Print results ---
    print("--- Candidate Rankings ---\n")
    ranked = sorted(enumerate(centrality_scores), key=lambda x: x[1], reverse=True)

    for rank, (idx, score) in enumerate(ranked):
        marker = " <-- BEST" if idx == best_idx else ""
        print(
            f"  #{rank + 1} (candidate {idx + 1}, centrality: {score:.4f}){marker}\n"
            f"     answer: \"{truncate(answers[idx], 80)}\""
        )

    print(f"\n--- Consensus Answer (candidate {best_idx + 1}) ---")
    print(f"Final Answer: {answers[best_idx]}")
    print("\n--- Full Response ---")
    print(candidates[best_idx])

    set_return(answers[best_idx])


if __name__ == "__main__":
    main()
