"""Text watermarking example using inferlib (Python).

Uses a green/red list approach where tokens are partitioned based on the
hash of the previous token, and green-listed tokens receive a probability
boost during sampling.  Uses Context.decode_step_dist() to obtain per-step
token distributions, then applies the watermark bias before sampling.
"""

import hashlib
import math
import random

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
        rng = random.Random(seed)

        green_list_size = round(len(ids) * self.gamma)
        indices = list(range(len(ids)))
        rng.shuffle(indices)
        green_set = set(indices[:green_list_size])

        exp_delta = math.exp(self.delta)
        watermarked = [
            p * exp_delta if i in green_set else p for i, p in enumerate(probs)
        ]

        total = sum(watermarked)
        if total > 0:
            watermarked = [p / total for p in watermarked]

        chosen_idx = random.choices(
            range(len(watermarked)), weights=watermarked, k=1
        )[0]
        self.previous_token = ids[chosen_idx]
        return ids[chosen_idx]


def main() -> None:
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
