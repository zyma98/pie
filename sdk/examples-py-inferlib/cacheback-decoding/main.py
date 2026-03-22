"""Cacheback decoding example using inferlib (Python).

Demonstrates speculative decoding with a cache-based drafter using n-gram
matching.  The CacheDrafter records token patterns from the previous context
and speculates future tokens.  The main model then verifies the speculated
tokens via Context.verify_draft(), accepting matches and rejecting mismatches
in a single batched forward pass with tree attention.
"""

from inference_bindings import (
    Context,
    Model,
    set_return,
)
from run_bindings import get_arguments


class LruRow:
    """A simple fixed-size LRU cache row."""

    def __init__(self, max_columns: int) -> None:
        self._items: list[tuple[int, ...]] = []
        self._max = max_columns

    def insert(self, item: tuple[int, ...]) -> None:
        if item in self._items:
            self._items.remove(item)
        elif len(self._items) >= self._max:
            self._items.pop()
        self._items.insert(0, item)

    @property
    def items(self) -> list[tuple[int, ...]]:
        return list(self._items)


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


class CacheDrafter:
    """Records token patterns and speculates future tokens via n-gram matching."""

    def __init__(
        self, n_prev: int, n_next: int, n_row: int, n_column: int
    ) -> None:
        self.n_prev = n_prev
        self.n_next = n_next
        self.n_row = n_row
        self.n_column = n_column
        self.table: list[tuple[tuple[int, ...], LruRow]] = []
        self.prev_window: list[int] = [0] * (n_prev + n_next - 1)

    def _update_cache(
        self, prev_tokens: tuple[int, ...], next_tokens: tuple[int, ...]
    ) -> None:
        for idx, (k, _) in enumerate(self.table):
            if k == prev_tokens:
                _, cache = self.table.pop(idx)
                cache.insert(next_tokens)
                self.table.insert(0, (prev_tokens, cache))
                return
        cache = LruRow(self.n_column)
        cache.insert(next_tokens)
        if len(self.table) >= self.n_row:
            self.table.pop()
        self.table.insert(0, (prev_tokens, cache))

    def update(self, context: list[int]) -> None:
        full = self.prev_window + context
        window_size = self.n_prev + self.n_next
        for i in range(len(full) - window_size + 1):
            prev = tuple(full[i : i + self.n_prev])
            nxt = tuple(full[i + self.n_prev : i + window_size])
            self._update_cache(prev, nxt)
        self.prev_window = full[-(self.n_prev + self.n_next - 1) :]

    def draft(self) -> tuple[list[int], list[int]]:
        positions = list(range(1, self.n_next + 1))
        trie = TrieForest(1)
        key = tuple(self.prev_window[-self.n_prev :])
        for k, cache in self.table:
            if k == key:
                for item in cache.items:
                    trie.insert(list(item), positions)
                break
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

    drafter = CacheDrafter(n_prev=1, n_next=2, n_row=256, n_column=4)
    drafter.update(ctx.get_token_ids())

    all_generated: list[int] = []
    num_per_step: list[int] = []

    print("Starting generation with speculative decoding...")

    while True:
        draft_tokens, draft_pos_ids = drafter.draft()
        accepted = ctx.verify_draft(draft_tokens, draft_pos_ids)

        drafter.update(accepted)
        num_per_step.append(len(accepted))
        all_generated.extend(accepted)

        if len(all_generated) >= max_tokens:
            break
        if any(
            all_generated[-len(seq):] == seq
            for seq in eos_tokens
            if seq
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
