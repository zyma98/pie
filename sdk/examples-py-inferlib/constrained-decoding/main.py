"""
Grammar-constrained decoding example using inferlib (Python).

Uses the inferlib-llguidance Wasm component (linked at runtime) to constrain
model outputs to valid structured formats (e.g. JSON) by masking invalid tokens
during sampling. Context.decode_step_dist() returns the full next-token
distribution, and the ConstrainedSampler picks the best grammar-valid token.
"""

from inference_bindings import (
    Context,
    Model,
    set_return,
)
from run_bindings import get_arguments
from sampler import ConstrainedSampler

JSON_GRAMMAR = r"""
?start: value
?value: object
        | array
        | string
        | SIGNED_NUMBER      -> number
        | "true"             -> true
        | "false"            -> false
        | "null"             -> null
array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : string ":" value
string : ESCAPED_STRING
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"""


def main() -> None:
    args = get_arguments()
    prompt = args.get(
        "prompt",
        "Where is the capital of France? "
        "Output in compact JSON text format and only the JSON object. "
        "Don't include any additional whitespace, newlines, quotes, or comments.",
    )
    grammar = args.get("grammar", JSON_GRAMMAR)
    max_tokens = int(args.get("max_tokens", "128"))

    model = Model.get_auto()
    model_name = model.get_name()

    if not (
        model_name.startswith("llama-3")
        or model_name.startswith("Qwen/Qwen3")
        or model_name.startswith("deepseek-r1-distill-qwen-2")
    ):
        print(
            f"Constrained decoding example is only implemented for "
            f"Llama 3, Qwen 3, and DeepSeek R1 Distill Qwen 2. Got: {model_name}"
        )
        return

    escape_non_printable = model_name.startswith(
        "Qwen/Qwen3"
    ) or model_name.startswith("deepseek-r1-distill-qwen-2")

    eot_token_id = None
    for eos_seq in model.eos_tokens():
        if len(eos_seq) == 1:
            eot_token_id = eos_seq[0]
            break

    if eot_token_id is None:
        print(f"No single EOS token found for model: {model_name}")
        return

    tokenizer = model.get_tokenizer()
    ctx = Context(model)

    sampler = ConstrainedSampler(
        tokenizer.get_vocabs(),
        tokenizer.get_special_tokens(),
        tokenizer.get_split_regex(),
        grammar,
        eot_token_id,
        escape_non_printable,
    )

    ctx.fill_system("You are a helpful, respectful and honest assistant.")
    ctx.fill_user(prompt)

    if model_name.startswith("llama-3"):
        ctx.fill("\n\n")
    elif model_name.startswith("Qwen/Qwen3"):
        ctx.fill("\n\n<think></think>\n\n")
    elif model_name.startswith("deepseek-r1-distill-qwen-2"):
        ctx.fill("\n</think>\n\n")

    eos_sequences = model.eos_tokens()

    generated_token_ids: list[int] = []
    while True:
        dist = ctx.decode_step_dist(1.0, None)
        token = sampler.sample(dist.ids, dist.probs)
        ctx.fill_token(token)
        generated_token_ids.append(token)

        if len(generated_token_ids) >= max_tokens:
            break
        if any(
            generated_token_ids[-len(seq):] == seq
            for seq in eos_sequences
            if len(seq) <= len(generated_token_ids)
        ):
            break

    output = tokenizer.detokenize(generated_token_ids)

    print(f"Output: {output}")

    if generated_token_ids:
        print(f"Tokens generated: {len(generated_token_ids)}")

    set_return(output)


if __name__ == "__main__":
    main()
