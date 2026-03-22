"""
Text completion example using inferlib (Python).

This example mirrors the Rust text-completion-inferlib example.
It uses the inferlib inference component for all heavy lifting
(context management, KV cache, forward passes).
"""

from inference_bindings import (
    Context,
    Model,
    ChatFormatter,
    SamplerConfig_TopP,
    StopConfig,
    set_return,
)
from run_bindings import get_arguments


def main() -> None:
    args = get_arguments()
    prompt = args.get("prompt", "Hello, world!")
    max_tokens = int(args.get("max_tokens", "256"))
    system_prompt = args.get("system", "You are a helpful assistant.")

    model = Model.get_auto()

    formatter = ChatFormatter(model.get_prompt_template())
    formatter.add_system(system_prompt)
    formatter.add_user(prompt)
    rendered = formatter.render(True, True)

    ctx = Context(model)
    ctx.fill(rendered)

    sampler = SamplerConfig_TopP((0.6, 0.95))
    stop_config = StopConfig(max_tokens=max_tokens, eos_sequences=model.eos_tokens())
    result = ctx.generate(sampler, stop_config)

    set_return(result)


if __name__ == "__main__":
    main()
