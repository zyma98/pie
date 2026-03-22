"""
Agent swarm example using inferlib (Python).

Implements a single agent worker in a multi-agent pipeline where each agent
has a specific role (idea generator, plot developer, character creator, or
dialogue writer) and passes work to the next agent via broadcast/subscribe.
"""

from dataclasses import dataclass

from inference_bindings import (
    Context,
    Model,
    SamplerConfig_Greedy,
    StopConfig,
    broadcast,
    set_return,
    subscribe,
)
from run_bindings import get_arguments


@dataclass
class AgentConfig:
    name: str
    system_message: str
    task_instruction: str
    section_header: str
    prev_topic: str | None
    next_topic: str | None


AGENT_CONFIGS: dict[str, AgentConfig] = {
    "idea_generator": AgentConfig(
        name="Story Idea Generator",
        system_message=(
            "You are an expert idea generator on a collaborative story-writing "
            "team. Your role is to create a compelling, one-sentence story concept."
        ),
        task_instruction=(
            "Based on the user's request, generate a single, captivating sentence "
            "that establishes the core conflict or mystery of a story."
        ),
        section_header="Concept",
        prev_topic=None,
        next_topic="concept_to_plot",
    ),
    "plot_developer": AgentConfig(
        name="Plot Developer",
        system_message=(
            "You are a master storyteller on a collaborative writing team. Your "
            "role is to expand a story concept into a structured plot outline."
        ),
        task_instruction=(
            "Read the provided story **Concept**. Your task is to write a brief "
            "plot outline with three distinct acts (Act 1: Setup, Act 2: "
            "Confrontation, Act 3: Resolution)."
        ),
        section_header="Plot Outline",
        prev_topic="concept_to_plot",
        next_topic="plot_to_chars",
    ),
    "character_creator": AgentConfig(
        name="Character Creator",
        system_message=(
            "You are an expert character designer on a collaborative writing team. "
            "Your role is to create a memorable protagonist and antagonist."
        ),
        task_instruction=(
            "Read the **Concept** and **Plot Outline**. Your task is to create a "
            "one-sentence description for a compelling protagonist and a formidable "
            "antagonist that fit the story."
        ),
        section_header="Characters",
        prev_topic="plot_to_chars",
        next_topic="chars_to_dialogue",
    ),
    "dialogue_writer": AgentConfig(
        name="Dialogue Writer",
        system_message=(
            "You are a skilled dialogue writer on a collaborative writing team. "
            "Your role is to write a key piece of dialogue."
        ),
        task_instruction=(
            "Read all the story elements. Your task is to write a single, "
            "impactful line of dialogue spoken by the protagonist during the "
            "story's climax."
        ),
        section_header="Climax Dialogue",
        prev_topic="chars_to_dialogue",
        next_topic=None,
    ),
}


def main() -> None:
    args = get_arguments()
    my_role = args.get("role")
    if not my_role or my_role not in AGENT_CONFIGS:
        print(
            f"Unknown or missing role: {my_role!r}. "
            f"Must be one of: {', '.join(AGENT_CONFIGS)}"
        )
        return

    group_id = int(args.get("group_id", "0"))
    tokens_per_step = int(args.get("tokens_per_step", "512"))

    model = Model.get_auto()

    if not model.get_name().startswith("llama-3"):
        print(
            "This example works with only non-thinking models. "
            "Please use Llama 3 models."
        )
        return

    eos_tokens = model.eos_tokens()
    config = AGENT_CONFIGS[my_role]

    if config.prev_topic is not None:
        accumulated = subscribe(f"{config.prev_topic}-{group_id}")
        user_prompt = (
            f"**Previous Story Elements:**\n---\n{accumulated}\n---\n\n"
            f"**Your Specific Task:**\n{config.task_instruction}"
        )
    else:
        accumulated = ""
        user_prompt = args.get(
            "prompt", "A story about day dreaming in a park"
        )

    ctx = Context(model)
    ctx.fill_system(config.system_message)
    ctx.fill_user(f"{user_prompt}\nPlease start with \"### {config.section_header}\"")

    stop_config = StopConfig(
        max_tokens=tokens_per_step, eos_sequences=eos_tokens
    )
    contribution = ctx.generate(SamplerConfig_Greedy(), stop_config)

    tokenizer = model.get_tokenizer()
    eos_strings = [tokenizer.detokenize(tokens) for tokens in eos_tokens]
    for eos_str in eos_strings:
        if contribution.endswith(eos_str):
            contribution = contribution[: -len(eos_str)]
            break

    new_accumulated = f"{accumulated}\n{contribution}".strip()

    if config.next_topic is not None:
        broadcast(f"{config.next_topic}-{group_id}", new_accumulated)
        print(f"Broadcasted story to channel: {config.next_topic}-{group_id}")
    else:
        print(f"Final story:\n{new_accumulated}")

    set_return(new_accumulated)


if __name__ == "__main__":
    main()
