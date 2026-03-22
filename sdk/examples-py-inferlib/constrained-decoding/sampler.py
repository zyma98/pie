"""A custom sampler that constrains token generation to match a Lark grammar.

Uses GrammarMatcher and TokenMask from inferlib-llguidance (linked at runtime
as a Wasm component) to compute token masks based on grammar state, ensuring
outputs are always syntactically valid. Called from the manual decode loop in
main.py via Context.decode_step_dist.

Mirrors the Rust ConstrainedSampler in
examples-inferlib/constrained-decoding/src/sampler.rs.
"""

from llguidance_bindings import GrammarMatcher


class ConstrainedSampler:
    def __init__(
        self,
        vocab: tuple[list[int], list[bytes]],
        special_tokens: tuple[list[int], list[bytes]],
        split_regex: str,
        grammar: str,
        eos_token_id: int,
        escape_non_printable: bool,
    ) -> None:
        vocab_ids, vocab_bytes = vocab
        special_token_ids, special_token_bytes = special_tokens

        self._matcher = GrammarMatcher(
            vocab_ids,
            vocab_bytes,
            special_token_ids,
            special_token_bytes,
            split_regex,
            grammar,
            eos_token_id,
            escape_non_printable,
        )
        self._eos_token_id = eos_token_id

    def sample(self, token_ids: list[int], probs: list[float]) -> int:
        """Pick the highest-probability token allowed by the grammar mask."""
        mask = self._matcher.compute_mask()
        if mask is None:
            return self._eos_token_id

        if mask.is_empty():
            return self._eos_token_id

        max_prob = float("-inf")
        best_token = None

        # Find the highest-probability token allowed by the grammar mask
        for i, token_id in enumerate(token_ids):
            if mask.is_allowed(token_id) and probs[i] > max_prob:
                max_prob = probs[i]
                best_token = token_id

        if best_token is None:
            fb = mask.first_bit_set()
            return fb if fb is not None else 0

        # Commit the chosen token to advance the parser state
        self._matcher.consume_token(best_token)
        return best_token
