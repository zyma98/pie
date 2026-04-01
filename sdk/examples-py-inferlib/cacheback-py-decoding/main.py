"""Cacheback decoding example using inferlib (Python).

Demonstrates speculative decoding with a cache-based drafter using n-gram
matching.  The cache table is provided by the inferlib-cacheback-py Python
library component which manages the two-level LRU cache, Trie-based draft
organization, and sliding-window state internally.
The main model verifies speculated tokens via Context.verify_draft(),
accepting matches and rejecting mismatches in a single batched forward pass
with tree attention.
"""

from cacheback_py_bindings import CacheTable
from inference_bindings import (
    Context,
    Model,
    set_return,
)
from run_bindings import get_arguments

LEADER_CAPACITY = 256
FOLLOWER_CAPACITY = 4
LEADER_LEN = 1
FOLLOWER_LEN = 2


def main() -> None:
    args = get_arguments()
    prompt = args.get("prompt", "Keep printing 'hello, world!' 100 times.")
    max_tokens = int(args.get("max_tokens", "256"))

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    tokenizer = model.get_tokenizer()

    ctx = Context(model)
    ctx.fill_system("You are a helpful, respectful and honest assistant.")
    ctx.fill_user(prompt)

    cache_table = CacheTable(LEADER_CAPACITY, FOLLOWER_CAPACITY, LEADER_LEN, FOLLOWER_LEN)

    all_generated: list[int] = []
    num_per_step: list[int] = []

    print("Starting generation with speculative decoding...")

    # Seed the cache table with prompt n-grams.  We cannot call
    # ctx.get_token_ids() before the first verify_draft because fill_system /
    # fill_user place tokens in a pending buffer that is only flushed to
    # token_ids during verify_draft.  So we run one iteration with an empty
    # draft first, then seed the cache with the now-available prompt token IDs.
    result = cache_table.draft()
    accepted = ctx.verify_draft(result.tokens, result.positions)
    cache_table.update(ctx.get_token_ids())
    cache_table.update(accepted)
    num_per_step.append(len(accepted))
    all_generated.extend(accepted)

    while True:
        if len(all_generated) >= max_tokens:
            break
        if any(
            all_generated[-len(seq) :] == seq for seq in eos_tokens if seq
        ):
            break

        result = cache_table.draft()
        accepted = ctx.verify_draft(result.tokens, result.positions)

        cache_table.update(accepted)
        num_per_step.append(len(accepted))
        all_generated.extend(accepted)

        if len(all_generated) >= max_tokens:
            break
        if any(
            all_generated[-len(seq) :] == seq for seq in eos_tokens if seq
        ):
            break

    output = tokenizer.detokenize(all_generated)

    print("Generation completed.")
    print(f"Output: {output!r}")

    if all_generated:
        print(f"Tokens generated: {len(all_generated)}")
        mean_accepted = sum(num_per_step) / len(num_per_step)
        print(f"Mean accepted tokens per step: {mean_accepted:.4f}")

    set_return(output)


if __name__ == "__main__":
    main()
