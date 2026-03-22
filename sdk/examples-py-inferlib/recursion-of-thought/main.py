"""
Recursion-of-Thought example using inferlib (Python).

The model recursively decides whether to solve a problem directly (leaf node)
or divide it into two independent subtasks (branch node).  Solutions from
subtasks are merged to produce the final answer.
"""

from __future__ import annotations

from typing import Generator

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

# A Task is a generator that yields GenerateFuture objects and returns a str.
# The driver sends back the completed string result for each yielded future.
Task = Generator[GenerateFuture, str, str]


def run_task(task: Task) -> str:
    """Drive a single Task generator to completion, blocking on each future."""
    try:
        future = next(task)
    except StopIteration as e:
        return e.value

    while True:
        pollable = future.pollable()
        pollable.block()
        del pollable
        result = future.get()
        if result is not None:
            try:
                future = task.send(result)
            except StopIteration as e:
                return e.value


def join_two_tasks(task1: Task, task2: Task) -> tuple[str, str]:
    """Drive two Task generators concurrently, polling their futures together."""
    futures: list[GenerateFuture | None] = [None, None]
    results: list[str | None] = [None, None]
    gens = [task1, task2]

    for i in range(2):
        try:
            futures[i] = next(gens[i])
        except StopIteration as e:
            results[i] = e.value

    while results[0] is None or results[1] is None:
        active: list[tuple[int, GenerateFuture]] = []
        for i in range(2):
            if results[i] is None and futures[i] is not None:
                active.append((i, futures[i]))
        if not active:
            break

        pollables = [f.pollable() for _, f in active]
        ready_indices = poll(pollables)
        del pollables

        for ri in ready_indices:
            idx, fut = active[ri]
            result = fut.get()
            if result is not None:
                try:
                    futures[idx] = gens[idx].send(result)
                except StopIteration as e:
                    results[idx] = e.value
                    futures[idx] = None

    return results[0], results[1]


def poll_flush(ctx: Context) -> None:
    """Block until a flush_async completes."""
    future = ctx.flush_async()
    if future is not None:
        pollable = future.pollable()
        pollable.block()
        del pollable
        del future

DIVIDE_PROMPT_TEMPLATE = (
    "Your task is to analyze the given problem and decide whether it can be solved "
    "directly or needs to be divided into smaller subproblems. If the problem is "
    "simple and can be solved immediately, provide the solution wrapped in "
    "`<leaf>THE ANSWER</leaf>`. If not, divide the problem into exactly two "
    "independent subtasks such that solving these subtasks and combining their "
    "solutions will lead to the solution of the original problem. Present the "
    "subtasks wrapped in `<branch>SUBTASK 1</branch>` and "
    "`<branch>SUBTASK 2</branch>`. Be concise and ensure the subtasks are distinct "
    "and solvable. Please also ensure that the description of the subtasks is clear "
    "and self-contained, that is, each subtask should be able to be solved "
    "independently of the other. One subtask should not depend on the result of the "
    "other subtask. Problem: {}"
)

SOLVE_PROMPT = (
    "Now, please solve the problem. Reason step-by-step. Make your response short."
)

MERGE_PROMPT = (
    "Now, please merge the two solutions into one. Make your response short."
)


def _strip_thinking(response: str) -> str:
    """Strip a <think>...</think> block from the start of a model response, if present."""
    if "</think>" in response:
        return response[response.index("</think>") + len("</think>"):]
    return response


def _parse_response(
    response: str,
) -> tuple[str | None, tuple[str, str] | None, str | None]:
    """Return (leaf_answer, branch_pair, error)."""
    response = _strip_thinking(response)
    if "<leaf>" in response and "</leaf>" in response:
        start = response.index("<leaf>") + len("<leaf>")
        end = response.index("</leaf>")
        return response[start:end].strip(), None, None

    branches: list[str] = []
    remaining = response
    while "<branch>" in remaining and "</branch>" in remaining:
        start = remaining.index("<branch>") + len("<branch>")
        end = remaining.index("</branch>")
        branches.append(remaining[start:end].strip())
        remaining = remaining[end + len("</branch>") :]

    if len(branches) == 2:
        return None, (branches[0], branches[1]), None

    return None, None, (
        f"Expected a <leaf> tag or exactly two <branch> tags, "
        f"but found {len(branches)} branches."
    )


def divide_and_conquer(
    ctx: Context,
    question: str,
    eos_tokens: list[list[int]],
    path: str,
    max_depth: int,
    max_tokens: int,
    verbose: bool,
) -> Task:
    stop_config = StopConfig(max_tokens=max_tokens, eos_sequences=eos_tokens)

    # Base case
    if len(path) >= max_depth:
        solve_ctx = ctx
        solve_ctx.fill_user(f"{SOLVE_PROMPT} {question}")
        future = solve_ctx.generate_async(SamplerConfig_Greedy(), stop_config)
        response = yield future
        if verbose:
            print(f"Reached max depth at path {path!r}")
            print(f"Response: {response.strip()}\n")
        return response

    # Recursive step
    if verbose:
        print(f"Analysing problem at path {path!r}")

    divide_ctx = ctx.fork()
    divide_prompt = DIVIDE_PROMPT_TEMPLATE.format(question)
    divide_ctx.fill_user(divide_prompt)
    future = divide_ctx.generate_async(SamplerConfig_Greedy(), stop_config)
    response = yield future

    if verbose:
        print(f"Response: {response.strip()}")

    leaf, branch_pair, error = _parse_response(response)

    if leaf is not None:
        if verbose:
            print(f"Leaf node found at path {path!r}")
            print(f"Response: {leaf.strip()}\n")
        return leaf

    if branch_pair is not None:
        sub1, sub2 = branch_pair
        if verbose:
            print(f"Branch node found at path {path!r}")
            print(f"Subtask 1: {sub1.strip()}")
            print(f"Subtask 2: {sub2.strip()}\n")

        gen1 = divide_and_conquer(
            ctx.fork(), sub1, eos_tokens, f"{path}l", max_depth, max_tokens, verbose
        )
        gen2 = divide_and_conquer(
            ctx.fork(), sub2, eos_tokens, f"{path}r", max_depth, max_tokens, verbose
        )

        if verbose:
            solution1 = run_task(gen1)
            solution2 = run_task(gen2)
        else:
            solution1, solution2 = join_two_tasks(gen1, gen2)

        if verbose:
            print(f"Merging solutions at path {path!r}")

        merge_ctx = ctx
        merge_prompt = (
            f"Subtask 1 solution: {solution1}\n"
            f"Subtask 2 solution: {solution2}\n{MERGE_PROMPT}"
        )
        merge_ctx.fill_user(merge_prompt)
        future = merge_ctx.generate_async(SamplerConfig_Greedy(), stop_config)
        response = yield future

        if verbose:
            print(f"Response: {response.strip()}\n")
        return response

    if error is not None:
        return f"Parsing Error: {error}"

    return "Error: Invalid response format from model."


def main() -> None:
    args = get_arguments()
    question = args.get(
        "question", args.get("q", "Please calculate the expression (42 + 3) * 5 / 15.")
    )
    max_depth = int(args.get("max-depth", args.get("d", "5")))
    max_tokens = int(args.get("max-tokens", args.get("t", "128")))
    verbose = bool(args.get("verbose", False))

    print("--- Initializing Model and Context ---")
    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    ctx = Context(model)

    ctx.fill_system("You are a helpful, respectful and honest assistant.")
    poll_flush(ctx)

    print("--- Starting Recursion-of-Thought (RoT) ---")
    print(f"Question: {question}")
    print(f"Max Depth: {max_depth}, Max Tokens: {max_tokens}")

    task = divide_and_conquer(
        ctx, question, eos_tokens, "", max_depth, max_tokens, verbose
    )
    solution = run_task(task)

    print(f"\n--- RoT Complete ---")
    print(f"\nFinal solution: {solution}")

    set_return(solution)


if __name__ == "__main__":
    main()
