"""
ReAct-style agent example using inferlib (Python).

Implements a ReAct agent that performs sequential Thought/Action/Observation
cycles with actual tool execution.  Replaces the Rust version's chrono and
evalexpr crates with Python's datetime and eval() built-ins.
"""

from datetime import date, datetime

from inference_bindings import (
    Context,
    Model,
    SamplerConfig_Greedy,
    StopConfig,
    set_return,
)
from run_bindings import get_arguments

SYSTEM_PROMPT = """\
You are a helpful assistant that understands how to break down a complex question into \
a series of steps. \
The following tools are available:

- `Calculator[expression]`: Evaluates a mathematical expression \
                            (e.g., "15 * 30", "100 / 2.5", "5 + 3 * 2").
- `CurrentDate[]`: Returns today's date in YYYY-MM-DD format.
- `DaysBetween[YYYY-MM-DD, YYYY-MM-DD]`: Calculates the number of days between \
                                         two dates (from first date to second date).
- `FinalAnswer[answer]`: Reports your final answer to the user's question.

Please respond with one tool use at a time, and don't nest tool calls.

The user's question might be complicated, so it may require multiple steps to answer. \
You will receive a history of the interactions with the tools so far. Use this history \
to reason about the next action to take. If you don't see a history, it means that the \
conversation has just started.

You need to answer next action to take, you must output your thoughts and the action \
to take. The format should be:

Thought: Your reasoning for the next action.
Action: The tool to use, in the format `ToolName[input]`.

When possible, please prefer using the tools that are available.

In the interaction history, you will see the results of the previous tool calls with \
the following format:

Observation: The result of the tool call.

As a reminder, you must respond only the next action to take and end the conversation, \
and use only one tool call at a time."""

USER_PROMPT = (
    "If I save $12.50 per day starting today, how much money "
    "will I have saved by the end of the year 2030?"
)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _extract_tool_input(text: str, tool_name: str) -> str | None:
    text = text.strip()
    if tool_name not in text:
        return None
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        return text[start + 1 : end]
    return None


def _execute_calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


def _execute_current_date(_input: str) -> str:
    today = date.today()
    return f"Today's date is: {today.isoformat()}"


def _execute_days_between(input_str: str) -> str:
    parts = [s.strip() for s in input_str.split(",")]
    if len(parts) != 2:
        return (
            "Error: DaysBetween requires exactly 2 dates separated by comma. "
            "Expected format: DaysBetween[YYYY-MM-DD, YYYY-MM-DD]"
        )

    try:
        date_from = datetime.strptime(parts[0], "%Y-%m-%d").date()
    except ValueError as e:
        return f"Error parsing start date '{parts[0]}': {e}. Expected format: YYYY-MM-DD"

    try:
        date_until = datetime.strptime(parts[1], "%Y-%m-%d").date()
    except ValueError as e:
        return f"Error parsing end date '{parts[1]}': {e}. Expected format: YYYY-MM-DD"

    days = (date_until - date_from).days
    if days > 0:
        return f"There are {days} days from {parts[0]} to {parts[1]}"
    elif days == 0:
        return f"Both dates are the same: {parts[0]}"
    else:
        return f"The date {parts[1]} is {-days} days before {parts[0]}"


def _execute_final_answer(answer: str) -> str:
    return f"Task completed. Final answer: {answer.strip()}"


def _parse_and_execute_action(text: str) -> tuple[str, str | None]:
    """Returns (observation_or_status, final_answer_or_none).

    Scans backwards to find the last Action line.
    """
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("Action:"):
            continue
        action_part = line[len("Action:") :].strip()

        inner = _extract_tool_input(action_part, "Calculator")
        if inner is not None:
            return _execute_calculator(inner), None

        inner = _extract_tool_input(action_part, "CurrentDate")
        if inner is not None:
            return _execute_current_date(inner), None

        inner = _extract_tool_input(action_part, "DaysBetween")
        if inner is not None:
            return _execute_days_between(inner), None

        inner = _extract_tool_input(action_part, "FinalAnswer")
        if inner is not None:
            answer = _execute_final_answer(inner)
            return answer, answer

    return "No action detected. Please use the format: Action: ToolName[input]", None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_arguments()
    num_function_calls = int(args.get("num_function_calls", "5"))
    tokens_between_calls = int(args.get("tokens_between_calls", "512"))

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    ctx = Context(model)

    ctx.fill_system(SYSTEM_PROMPT)
    ctx.fill_user(f"{USER_PROMPT} What is the next step to solve this problem?")

    stop_config = StopConfig(
        max_tokens=tokens_between_calls, eos_sequences=eos_tokens
    )

    final_answer: str | None = None

    for _ in range(num_function_calls):
        response = ctx.generate(SamplerConfig_Greedy(), stop_config)
        observation, answer = _parse_and_execute_action(response)

        if answer is not None:
            final_answer = answer
            break

        ctx.fill_user(
            f"Observation: {observation}\n What is the next step to solve this problem?"
        )

    print(f"Full context: {ctx.get_text()}")

    if final_answer is not None:
        print(f"Final answer: {final_answer}")
    else:
        print("No final answer found.")

    set_return(final_answer or "No final answer found.")


if __name__ == "__main__":
    main()
