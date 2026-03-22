"""
Windowed attention example using inferlib (Python).

Generates text using a simple sliding window for KV cache management.
Only the most recent ``window_size`` tokens are kept in the KV cache;
older tokens are masked and evicted.
"""

from inference_bindings import (
    Context,
    Model,
    SamplerConfig,
    SamplerConfig_Greedy,
    StopConfig,
    Tokenizer,
    set_return,
)
from run_bindings import get_arguments


def _check_stop(generated: list[int], stop_config: StopConfig) -> bool:
    if len(generated) >= stop_config.max_tokens:
        return True
    for eos in stop_config.eos_sequences:
        if generated[-len(eos) :] == eos:
            return True
    return False


def generate_with_sliding_window(
    ctx: Context,
    sampler: SamplerConfig,
    stop_config: StopConfig,
    tokenizer: Tokenizer,
    window_size: int,
) -> str:
    generated_token_ids: list[int] = []

    while True:
        next_token_id = ctx.decode_step(sampler)
        ctx.fill_token(next_token_id)
        generated_token_ids.append(next_token_id)

        if _check_stop(generated_token_ids, stop_config):
            break

        committed_len = len(ctx.get_token_ids())
        if committed_len > window_size:
            evict_end = committed_len - window_size
            ctx.mask_token_range(1, evict_end, True)
            ctx.drop_masked_kv_pages()

    return tokenizer.detokenize(generated_token_ids)


def main() -> None:
    args = get_arguments()
    prompt = args.get("prompt", "Explain LLM decoding process in ELI5.")
    max_tokens = int(args.get("max_tokens", "512"))
    window_size = int(args.get("window_size", "32"))

    model = Model.get_auto()
    tokenizer = model.get_tokenizer()

    ctx = Context(model)
    ctx.fill_system("You are a helpful, respectful and honest assistant.")
    ctx.fill_user(prompt)

    sampler = SamplerConfig_Greedy()
    stop_config = StopConfig(
        max_tokens=max_tokens, eos_sequences=model.eos_tokens()
    )

    print(f"Starting generation with Windowed Attention (window_size={window_size})")

    output = generate_with_sliding_window(
        ctx, sampler, stop_config, tokenizer, window_size
    )

    output_token_ids = tokenizer.tokenize(output)
    print(f"\n--- Output ---\n{output}\n--------------")
    print(f"Tokens generated: {len(output_token_ids)}")

    set_return(output)


if __name__ == "__main__":
    main()
