"""
Jacobi decoding example using inferlib (Python).

Implements the Parallel Jacobi Decoding (PJD) algorithm for speculative
generation.  Uses the raw Queue and ForwardPass APIs, managing KV cache pages
and position tracking manually.
"""

from inference_bindings import (
    ChatFormatter,
    Model,
    Queue,
    Tokenizer,
    set_return,
)
from run_bindings import get_arguments


def _grow_kv_pages(
    queue: "Queue",
    kv_page_ptrs: list[int],
    kv_page_last_len: int,
    kv_page_size: int,
    num_tokens: int,
) -> tuple[list[int], int]:
    if num_tokens == 0:
        return kv_page_ptrs, kv_page_last_len
    current_total = (
        0 if not kv_page_ptrs
        else (len(kv_page_ptrs) - 1) * kv_page_size + kv_page_last_len
    )
    new_total = current_total + num_tokens
    new_num_pages = (new_total + kv_page_size - 1) // kv_page_size
    pages_to_add = new_num_pages - len(kv_page_ptrs)
    if pages_to_add > 0:
        new_pages = queue.allocate_kv_pages(pages_to_add)
        kv_page_ptrs = kv_page_ptrs + list(new_pages)
    new_last_len = new_total % kv_page_size
    if new_last_len == 0:
        new_last_len = kv_page_size
    return kv_page_ptrs, new_last_len


def _shrink_kv_pages(
    queue: "Queue",
    kv_page_ptrs: list[int],
    kv_page_last_len: int,
    kv_page_size: int,
    num_tokens: int,
) -> tuple[list[int], int]:
    if num_tokens == 0:
        return kv_page_ptrs, kv_page_last_len
    current_total = (
        0 if not kv_page_ptrs
        else (len(kv_page_ptrs) - 1) * kv_page_size + kv_page_last_len
    )
    new_total = max(0, current_total - num_tokens)
    if new_total == 0:
        queue.deallocate_kv_pages(kv_page_ptrs)
        return [], 0
    new_num_pages = (new_total + kv_page_size - 1) // kv_page_size
    pages_to_remove = len(kv_page_ptrs) - new_num_pages
    if pages_to_remove > 0:
        removed = kv_page_ptrs[new_num_pages:]
        kv_page_ptrs = kv_page_ptrs[:new_num_pages]
        queue.deallocate_kv_pages(removed)
    new_last_len = new_total % kv_page_size
    if new_last_len == 0:
        new_last_len = kv_page_size
    return kv_page_ptrs, new_last_len


def _causal_mask(num_total_tokens: int, num_input_tokens: int) -> list[list[int]]:
    offset = num_total_tokens - num_input_tokens
    return [[offset + i + 1] for i in range(num_input_tokens)]


def _check_stop(
    tokens: list[int], max_len: int, eos_sequences: list[list[int]]
) -> bool:
    if len(tokens) >= max_len:
        return True
    for eos in eos_sequences:
        if len(tokens) >= len(eos) and tokens[-len(eos) :] == eos:
            return True
    return False


def generate_with_pjd(
    queue: "Queue",
    tokenizer: "Tokenizer",
    prompt_tokens: list[int],
    gamma: int,
    unk_token_id: int,
    max_tokens: int,
    eos_sequences: list[list[int]],
    kv_page_size: int,
) -> tuple[str, int]:
    all_generated: list[int] = []
    num_steps = 0

    kv_page_ptrs: list[int] = []
    kv_page_last_len = 0

    token_ids: list[int] = []
    position_ids: list[int] = []

    batch_tokens = prompt_tokens + [unk_token_id] * gamma

    while True:
        if _check_stop(all_generated, max_tokens, eos_sequences):
            break

        batch_len = len(batch_tokens)

        pos_offset = (position_ids[-1] + 1) if position_ids else 0
        batch_positions = list(range(pos_offset, pos_offset + batch_len))

        kv_page_ptrs, kv_page_last_len = _grow_kv_pages(
            queue, kv_page_ptrs, kv_page_last_len, kv_page_size, batch_len
        )

        total_ctx_len = len(token_ids) + batch_len
        masks = _causal_mask(total_ctx_len, batch_len)

        sample_indices = list(range(batch_len - gamma - 1, batch_len))

        fp = queue.create_forward_pass()
        fp.input_tokens(batch_tokens, batch_positions)
        fp.kv_cache(kv_page_ptrs, kv_page_last_len)
        fp.attention_mask(masks)
        fp.output_tokens(sample_indices, 0.0)
        result = fp.execute()
        sampled_tokens = list(result.tokens) if result.tokens else []

        speculated = batch_tokens[batch_len - gamma :]

        accepted = [sampled_tokens[0]]
        rejected: list[int] = []
        for i in range(gamma):
            if sampled_tokens[i] == speculated[i]:
                accepted.append(sampled_tokens[i + 1])
            else:
                rejected = sampled_tokens[i + 1 :]
                break

        kv_page_ptrs, kv_page_last_len = _shrink_kv_pages(
            queue, kv_page_ptrs, kv_page_last_len, kv_page_size, len(rejected)
        )

        token_ids.extend(batch_tokens[: batch_len - gamma])
        token_ids.extend(accepted[: len(accepted) - 1])
        position_ids.extend(batch_positions[: batch_len - gamma])
        position_ids.extend(
            batch_positions[batch_len - gamma : batch_len - len(rejected)]
        )

        all_generated.extend(accepted)

        batch_tokens = list(accepted)
        add_unk = gamma - len(rejected)
        batch_tokens.extend(rejected)
        batch_tokens.extend([unk_token_id] * add_unk)

        num_steps += 1

    return tokenizer.detokenize(all_generated), num_steps


def main() -> None:
    args = get_arguments()
    max_tokens = int(args.get("max_tokens", "512"))
    speculation_length = int(args.get("speculation_length", "5"))
    prompt = args.get("prompt", "Explain the LLM decoding process ELI5.")

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    tokenizer = model.get_tokenizer()
    kv_page_size = model.get_kv_page_size()

    unk_token_id = eos_tokens[0][0]

    template = model.get_prompt_template()
    formatter = ChatFormatter(template)
    formatter.add_system("You are a helpful, respectful and honest assistant.")
    formatter.add_user(prompt)
    formatted = formatter.render(True, True)
    prompt_tokens = tokenizer.tokenize(formatted)

    queue = Queue.from_model_name(model.get_name())

    print(
        f"Starting generation with Parallel Jacobi Decoding "
        f"(speculation length = {speculation_length})..."
    )

    output, num_steps = generate_with_pjd(
        queue,
        tokenizer,
        prompt_tokens,
        speculation_length,
        unk_token_id,
        max_tokens,
        eos_tokens,
        kv_page_size,
    )

    output_token_ids = tokenizer.tokenize(output)
    print(f"\n--- Output ---\n{output}\n--------------")
    print(
        f"Tokens generated: {len(output_token_ids)}, "
        f"Mean accepted tokens per step: {len(output_token_ids) / num_steps:.4f}"
        if num_steps > 0
        else "No steps taken."
    )

    set_return(output)


if __name__ == "__main__":
    main()
