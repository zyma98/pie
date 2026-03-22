"""
Text completion with greedy (deterministic) sampling (Python).

Identical to text-completion but uses greedy decoding instead of top-p
sampling, guaranteeing reproducible output for the same prompt.
"""

from inference_bindings import (
    Context,
    Model,
    SamplerConfig_Greedy,
    StopConfig,
    set_return,
)
from run_bindings import get_arguments


def main() -> None:
    args = get_arguments()
    prompt = args.get(
        "prompt", args.get("p", "Explain what makes a good unit test in three concise points.")
    )
    max_tokens = int(args.get("max-tokens", args.get("n", "256")))
    system_message = args.get(
        "system", args.get("s", "You are a helpful, respectful and honest assistant.")
    )

    model = Model.get_auto()
    ctx = Context(model)

    ctx.fill_system(system_message)
    ctx.fill_user(prompt)

    stop_config = StopConfig(max_tokens=max_tokens, eos_sequences=model.eos_tokens())
    result = ctx.generate(SamplerConfig_Greedy(), stop_config)

    print(f"Output: {result!r}")

    set_return(result)


if __name__ == "__main__":
    main()
