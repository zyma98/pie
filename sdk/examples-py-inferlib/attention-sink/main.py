"""
Attention sink example using inferlib (Python).

Implements a sliding-window attention mechanism (attention sink) to keep the
KV cache size bounded during generation.  Maintains an initial "sink" of
tokens plus a sliding window of recent tokens, masking out tokens in between.
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


def generate_with_attention_sink(
    ctx: Context,
    sampler: SamplerConfig,
    stop_config: StopConfig,
    tokenizer: Tokenizer,
    attention_sink_initial_size: int,
    attention_sink_window_size: int,
) -> str:
    generated_token_ids: list[int] = []
    max_cache_size = attention_sink_initial_size + attention_sink_window_size

    while True:
        next_token_id = ctx.decode_step(sampler)
        ctx.fill_token(next_token_id)
        generated_token_ids.append(next_token_id)

        if _check_stop(generated_token_ids, stop_config):
            break

        committed_len = len(ctx.get_token_ids())
        if committed_len > max_cache_size:
            num_to_evict = committed_len - max_cache_size
            evict_start = attention_sink_initial_size
            evict_end = attention_sink_initial_size + num_to_evict
            ctx.mask_token_range(evict_start, evict_end, True)
            ctx.drop_masked_kv_pages()

    return tokenizer.detokenize(generated_token_ids)


def main() -> None:
    args = get_arguments()
    max_tokens = int(args.get("max_tokens", "512"))
    sink_size = int(args.get("sink_size", "64"))
    sink_window = int(args.get("sink_window", "32"))
    prompt = args.get("prompt", "Explain LLM decoding process in ELI5.")

    model = Model.get_auto()
    tokenizer = model.get_tokenizer()
    eos_tokens = model.eos_tokens()

    ctx = Context(model)
    ctx.fill_system("You are a helpful, respectful and honest assistant.")
    ctx.fill_user(prompt)

    sampler = SamplerConfig_Greedy()
    stop_config = StopConfig(max_tokens=max_tokens, eos_sequences=eos_tokens)

    output = generate_with_attention_sink(
        ctx, sampler, stop_config, tokenizer, sink_size, sink_window
    )

    output_token_ids = tokenizer.tokenize(output)
    print(f"\n--- Output ---\n{output}\n--------------")
    print(f"Tokens generated: {len(output_token_ids)}")

    set_return(output)


if __name__ == "__main__":
    main()
