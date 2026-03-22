"""
Recursion-of-Thought example using inferlib (Python).

The model recursively decides whether to solve a problem directly (leaf node)
or divide it into two independent subtasks (branch node).  Solutions from
subtasks are merged to produce the final answer.
"""

from inference_bindings import (
    Context,
    GenerateFuture,
    Model,
    SamplerConfig_Greedy,
    StopConfig,
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


def _parse_response(
    response: str,
) -> tuple[str | None, tuple[str, str] | None, str | None]:
    """Return (leaf_answer, branch_pair, error)."""
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
) -> str:
    stop_config = StopConfig(max_tokens=max_tokens, eos_sequences=eos_tokens)

    # Base case
    if len(path) >= max_depth:
        solve_ctx = ctx
        solve_ctx.fill_user(f"{SOLVE_PROMPT} {question}")
        future = solve_ctx.generate_async(SamplerConfig_Greedy(), stop_config)
        response = poll_generate(future)
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
    response = poll_generate(future)

    if verbose:
        print(f"Response: {response.strip()}")

    leaf, branch_pair, error = _parse_response(response)

    if leaf is not None:
        if verbose:
            print(f"Leaf node found at path {path!r}")
            print(f"Response: {leaf.strip()}\n")
        return leaf

    if branch_pair is not None:
        task1, task2 = branch_pair
        if verbose:
            print(f"Branch node found at path {path!r}")
            print(f"Subtask 1: {task1.strip()}")
            print(f"Subtask 2: {task2.strip()}\n")

        if verbose:
            solution1 = divide_and_conquer(
                ctx.fork(), task1, eos_tokens, f"{path}l", max_depth, max_tokens, verbose
            )
            solution2 = divide_and_conquer(
                ctx.fork(), task2, eos_tokens, f"{path}r", max_depth, max_tokens, verbose
            )
        else:
            solution1 = divide_and_conquer(
                ctx.fork(), task1, eos_tokens, f"{path}l", max_depth, max_tokens, verbose
            )
            solution2 = divide_and_conquer(
                ctx.fork(), task2, eos_tokens, f"{path}r", max_depth, max_tokens, verbose
            )

        if verbose:
            print(f"Merging solutions at path {path!r}")

        merge_ctx = ctx
        merge_prompt = (
            f"Subtask 1 solution: {solution1}\n"
            f"Subtask 2 solution: {solution2}\n{MERGE_PROMPT}"
        )
        merge_ctx.fill_user(merge_prompt)
        future = merge_ctx.generate_async(SamplerConfig_Greedy(), stop_config)
        response = poll_generate(future)

        if verbose:
            print(f"Response: {response.strip()}\n")
        return response

    if error is not None:
        return f"Parsing Error: {error}"

    return "Error: Invalid response format from model."


def main() -> None:
    args = get_arguments()
    question = args.get(
        "question", "Please calculate the expression (42 + 3) * 5 / 15."
    )
    max_depth = int(args.get("max_depth", "5"))
    max_tokens = int(args.get("max_tokens", "128"))
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

    solution = divide_and_conquer(
        ctx, question, eos_tokens, "", max_depth, max_tokens, verbose
    )

    print(f"\n--- RoT Complete ---")
    print(f"\nFinal solution: {solution}")

    set_return(solution)


if __name__ == "__main__":
    main()
