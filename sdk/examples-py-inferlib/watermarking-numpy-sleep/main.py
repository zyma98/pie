"""Text watermarking example using inferlib (Python) with numpy.

Uses a green/red list approach where tokens are partitioned based on the
hash of the previous token, and green-listed tokens receive a probability
boost during sampling.  Uses Context.decode_step_dist() to obtain per-step
token distributions, then applies the watermark bias before sampling.

Mathematical operations in the sampler use numpy instead of pure-Python
loops and list comprehensions.
"""

import time
import hashlib

import numpy as np

from inference_bindings import (
    Context,
    Model,
    set_return,
)
from run_bindings import get_arguments


class WatermarkSampler:
    """Injects a watermark by partitioning the vocabulary into green/red lists."""

    def __init__(self, gamma: float, delta: float) -> None:
        assert 0.0 <= gamma <= 1.0, "gamma must be between 0.0 and 1.0"
        self.gamma = gamma
        self.delta = delta
        self.previous_token: int | None = None

    def _get_seed(self) -> int:
        if self.previous_token is not None:
            return int(
                hashlib.sha256(
                    str(self.previous_token).encode()
                ).hexdigest(),
                16,
            ) % (2**32)
        return 0

    def sample(self, ids: list[int], probs: list[float]) -> int:
        if not ids:
            self.previous_token = 0
            return 0

        seed = self._get_seed()
        rng = np.random.default_rng(seed)

        n = len(ids)
        green_list_size = round(n * self.gamma)

        green_indices = rng.permutation(n)[:green_list_size]
        mask = np.zeros(n, dtype=bool)
        mask[green_indices] = True

        probs_arr = np.asarray(probs, dtype=np.float64)
        exp_delta = np.exp(self.delta)
        watermarked = np.where(mask, probs_arr * exp_delta, probs_arr)

        total = watermarked.sum()
        if total > 0:
            watermarked /= total

        chosen_idx = rng.choice(n, p=watermarked)
        self.previous_token = ids[chosen_idx]
        return ids[chosen_idx]


def main() -> None:
    time.sleep(10000)

    args = get_arguments()
    prompt = args.get("prompt", "Explain the LLM decoding process ELI5.")
    max_tokens = int(args.get("max_tokens", "256"))

    model = Model.get_auto()
    tokenizer = model.get_tokenizer()
    eos_sequences = model.eos_tokens()

    ctx = Context(model)
    watermark_sampler = WatermarkSampler(0.5, 0.0)

    ctx.fill_system("You are a helpful, respectful and honest assistant.")
    ctx.fill_user(prompt)

    generated_token_ids: list[int] = []
    while True:
        dist = ctx.decode_step_dist(1.0, None)
        token = watermark_sampler.sample(dist.ids, dist.probs)
        ctx.fill_token(token)
        generated_token_ids.append(token)

        if len(generated_token_ids) >= max_tokens:
            break
        if any(
            generated_token_ids[-len(seq):] == seq
            for seq in eos_sequences
            if seq
        ):
            break

    text = tokenizer.detokenize(generated_token_ids)
    print(f"Output: {text!r}")

    if generated_token_ids:
        print(f"Tokens generated: {len(generated_token_ids)}")

    set_return(text)


if __name__ == "__main__":
    main()
