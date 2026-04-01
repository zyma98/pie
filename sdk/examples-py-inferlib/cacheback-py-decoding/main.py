"""Cacheback decoding example using inferlib (Python).

Demonstrates speculative decoding with a cache-based drafter using n-gram
matching.  The cache table is provided by the inferlib-cacheback-py Python
library component; the Trie-based draft organization logic lives here locally.
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

LEADER_LEN = 1
FOLLOWER_LEN = 2
LEADER_CAPACITY = 256
FOLLOWER_CAPACITY = 4


class TrieNode:
    def __init__(self, position: int, token: int) -> None:
        self.children: list["TrieNode"] = []
        self.token = token
        self.position = position


class TrieForest:
    def __init__(self, root_position: int) -> None:
        self.roots: list[TrieNode] = []
        self.root_position = root_position

    def insert(self, tokens: list[int], positions: list[int]) -> None:
        if not tokens or not positions or positions[0] != self.root_position:
            return
        nodes = self.roots
        for tok, pos in zip(tokens, positions):
            found = None
            for n in nodes:
                if n.position == pos and n.token == tok:
                    found = n
                    break
            if found is None:
                found = TrieNode(pos, tok)
                nodes.append(found)
            nodes = found.children

    def linearize(self) -> tuple[list[int], list[int]]:
        tokens: list[int] = []
        positions: list[int] = []

        def dfs(node: TrieNode) -> None:
            tokens.append(node.token)
            positions.append(node.position)
            for child in node.children:
                dfs(child)

        for root in self.roots:
            dfs(root)
        return tokens, positions


def update_cache(
    cache_table: CacheTable,
    prev_window: list[int],
    context: list[int],
) -> None:
    window_len = LEADER_LEN + FOLLOWER_LEN - 1
    full = prev_window + context
    cache_table.update_cache(full)
    prev_window[:] = full[-window_len:]


def draft(
    cache_table: CacheTable, prev_window: list[int]
) -> tuple[list[int], list[int]]:
    positions = list(range(1, FOLLOWER_LEN + 1))
    trie = TrieForest(1)

    key = prev_window[-LEADER_LEN:]
    drafts = cache_table.get_draft_tokens(key)
    for d in drafts:
        trie.insert(list(d), positions)

    return trie.linearize()


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
    prev_window = [0] * (LEADER_LEN + FOLLOWER_LEN - 1)
    update_cache(cache_table, prev_window, ctx.get_token_ids())

    all_generated: list[int] = []
    num_per_step: list[int] = []

    print("Starting generation with speculative decoding...")

    while True:
        draft_tokens, draft_pos_ids = draft(cache_table, prev_window)
        accepted = ctx.verify_draft(draft_tokens, draft_pos_ids)

        update_cache(cache_table, prev_window, accepted)
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
