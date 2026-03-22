"""
CodeACT-style agent example using inferlib (Python).

Implements a CodeACT agent that generates and executes JavaScript code to
solve problems, using the js-engine library component for execution.
The agent iteratively writes code, receives execution results, and
continues until it arrives at the final answer.
"""

from inference_bindings import (
    Context,
    Model,
    SamplerConfig_Greedy,
    StopConfig,
    set_return,
)
from run_bindings import get_arguments
from js_engine_bindings import execute as execute_js_code

SYSTEM_PROMPT = """\
You are CodeACT, a highly intelligent AI assistant that solves problems by writing \
and executing JavaScript code step by step.

## Interaction Format

You will be given a task to solve, and you need to respond with the code that carries out \
the next step to solve the task. You may also receive a history of previous steps and their \
execution results reported by the user.

If you receive a history of previous steps and their execution results, it will be \
formatted as follows:
Code execution result: [Execution result here]

If you don't receive a history of previous steps and their execution results, it means that \
the conversation has just started. You must generate the code for the first step to solve \
the task.

You must generate the code for the NEXT STEP ONLY. Do not repeat previous steps or generate \
multiple code blocks at once. Respond with the following format:

Thought: Your reasoning about what to do next based on the history.
```javascript
// JavaScript code for this step only
```

When you have the final answer and no more code needs to be executed, respond with:

Thought: I have the answer.
Final Answer: [Your final answer here]

Important Notes:

- Each code execution is stateless - you cannot reference variables from previous executions.
- If you need helper functions, you must redefine them in each code block.
- The last expression in your code block will be returned as the result.
- Keep each code block focused on a single step of your solution.

Reminder: You must respond with the code for the NEXT STEP ONLY. Do not repeat previous \
steps or generate multiple code blocks at once."""

USER_PROMPT = "Calculate the sum of the first 10 prime numbers."


def _extract_js_code(text: str) -> str | None:
    """Extract the last ```javascript ... ``` code block from text."""
    start_marker = "```javascript"
    end_marker = "```"
    start = text.rfind(start_marker)
    if start == -1:
        return None
    code_start = start + len(start_marker)
    end = text.find(end_marker, code_start)
    if end == -1:
        return None
    return text[code_start:end].strip()


def _extract_final_answer(text: str) -> str:
    """Extract the last 'Final Answer: ...' line from text."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("Final Answer:"):
            return line[len("Final Answer:") :].strip()
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1].strip() if lines else "Unknown"


def main() -> None:
    args = get_arguments()
    num_function_calls = int(args.get("num_function_calls", "5"))
    tokens_between_calls = int(args.get("tokens_between_calls", "512"))

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    ctx = Context(model)

    ctx.fill_system(SYSTEM_PROMPT)
    ctx.fill_user(f"{USER_PROMPT}\n\nWhat is the first step?")

    stop_config = StopConfig(
        max_tokens=tokens_between_calls, eos_sequences=eos_tokens
    )

    final_answer: str | None = None

    for _ in range(num_function_calls):
        response = ctx.generate(SamplerConfig_Greedy(), stop_config)

        js_code = _extract_js_code(response)
        if js_code is not None:
            observation = execute_js_code(js_code)
            ctx.fill_user(
                f"Code execution result: {observation}\n\nWhat is the next step?"
            )
        else:
            final_answer = _extract_final_answer(response)
            break

    print(f"Full context: {ctx.get_text()}")

    if final_answer is not None:
        print(f"Final answer: {final_answer}")
    else:
        print("No final answer found within the iteration limit.")

    set_return(final_answer or "No final answer found.")


if __name__ == "__main__":
    main()
