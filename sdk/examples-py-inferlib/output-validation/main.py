"""
Output validation example using inferlib (Python).

Demonstrates how to evaluate the likelihood of different candidate outputs
given a context.  Uses the low-level ForwardPass API to obtain next-token
distributions.
"""

import math

from inference_bindings import (
    Context,
    Model,
    Queue,
    set_return,
)
from run_bindings import get_arguments


def validate_outputs(
    model: "Model",
    ctx: "Context",
    candidates: list[str],
) -> list[tuple[str, float]]:
    """Calculate normalized probabilities for each candidate string."""
    tokenizer = model.get_tokenizer()
    queue = Queue.from_model_name(model.get_name())
    log_probs: list[float] = []

    for candidate in candidates:
        candidate_ctx = ctx.fork()
        candidate_tokens = tokenizer.tokenize(candidate)
        current_log_prob = 0.0

        for token_id in candidate_tokens:
            candidate_ctx.flush()

            fp = queue.create_forward_pass()
            committed = candidate_ctx.get_token_ids()
            last_token = committed[-1]
            position = len(committed) - 1

            fp.input_tokens([last_token], [position])
            fp.kv_cache(
                candidate_ctx.get_kv_page_ptrs(),
                candidate_ctx.get_kv_page_last_len(),
            )
            fp.output_distributions([0], 1.0, None)
            result = fp.execute()

            if result.distributions is not None and result.distributions:
                dist = result.distributions[0]
                found = False
                for i, tid in enumerate(dist.ids):
                    if tid == token_id:
                        prob = dist.probs[i]
                        if prob > 0.0:
                            current_log_prob += math.log(prob)
                        else:
                            current_log_prob = -1000.0
                        found = True
                        break
                if not found:
                    current_log_prob = -1000.0
                    break

            candidate_ctx.fill_token(token_id)

        log_probs.append(current_log_prob)

    # Normalize using softmax
    max_lp = max(log_probs) if log_probs else float("-inf")

    if math.isinf(max_lp) and max_lp < 0:
        uniform = 1.0 / len(candidates)
        return [(c, uniform) for c in candidates]

    probs = [math.exp(lp - max_lp) for lp in log_probs]
    total = sum(probs)
    return [(c, p / total) for c, p in zip(candidates, probs)]


def main() -> None:
    args = get_arguments()

    model = Model.get_auto()
    ctx = Context(model)

    if not model.get_name().startswith("llama-3"):
        print(
            f"Output validation example is only implemented for Llama 3 models. "
            f"Got: {model.get_name()}"
        )
        return

    prompt = "The name of the person in the report is "
    ctx.fill("<|begin_of_text|>")
    ctx.fill(
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert at information extraction.<|eot_id|>"
    )
    ctx.fill(
        "<|start_header_id|>user<|end_header_id|>\n\n"
        'From the sentence "The financial report was prepared by David Chen.", '
        "extract the person's name.<|eot_id|>"
    )
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n")
    ctx.fill(prompt)
    ctx.flush()

    candidates = ["John Smith", "Mary Anne", "David Chen", "Chen David"]

    print(f"--- Context ---\n'{prompt}'\n\n--- Candidates ---")
    for c in candidates:
        print(f"- {c}")

    results = validate_outputs(model, ctx, candidates)

    print("\n--- Validation Results ---")
    for candidate, probability in results:
        print(f"- Candidate: {candidate:<12} | Probability: {probability * 100:.4f}%")

    set_return(
        "; ".join(f"{c}: {p * 100:.2f}%" for c, p in results)
    )


if __name__ == "__main__":
    main()
