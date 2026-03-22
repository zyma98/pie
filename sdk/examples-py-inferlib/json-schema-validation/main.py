"""
JSON Schema-validated generation with grammar-constrained decoding (Python).

Combines two layers of validation:
1. A Lark grammar via inferlib-llguidance ensures every generated token
   produces syntactically valid JSON (no parse errors possible).
2. The inferlib-schema component validates the JSON against a specific
   schema. When schema validation fails, the errors are fed back to the
   model and it regenerates -- still grammar-constrained.
"""

from inference_bindings import (
    Context,
    Model,
    set_return,
)
from llguidance_bindings import ConstrainedSampler
from run_bindings import get_arguments
from schema_bindings import SchemaValidator

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

PERSON_SCHEMA = """{
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1
        },
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 150
        },
        "email": {
            "type": "string"
        },
        "skills": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 1
        },
        "address": {
            "type": "object",
            "properties": {
                "city": { "type": "string" },
                "country": { "type": "string" }
            },
            "required": ["city", "country"]
        }
    },
    "required": ["name", "age", "email", "skills", "address"]
}"""

SYSTEM_PROMPT = (
    "You are a helpful assistant that generates structured data. When asked to produce "
    "JSON, you must output ONLY the raw JSON object with no additional text, markdown "
    "fences, or explanation. The JSON must be compact (no unnecessary whitespace). "
    "If you receive validation errors, fix exactly those issues and output the corrected "
    "JSON object."
)


def build_initial_prompt(user_prompt: str, schema: str) -> str:
    return (
        f"{user_prompt}\n\nThe output must conform to this JSON Schema:\n{schema}\n\n"
        "Output only the JSON object, nothing else."
    )


def build_retry_prompt(errors: str) -> str:
    return (
        f"The JSON you produced has schema validation errors:\n{errors}\n\n"
        "Please fix these errors and output only the corrected JSON object, nothing else."
    )


def generate_json(
    ctx: Context,
    sampler: ConstrainedSampler,
    tokenizer,
    eos_sequences: list[list[int]],
    max_tokens: int,
) -> str:
    generated_token_ids: list[int] = []
    while True:
        dist = ctx.decode_step_dist(1.0, None)
        token = sampler.sample(dist.ids, dist.probs)
        ctx.fill_token(token)
        generated_token_ids.append(token)

        if len(generated_token_ids) >= max_tokens:
            break

        for seq in eos_sequences:
            if seq and generated_token_ids[-len(seq):] == seq:
                generated_token_ids = generated_token_ids[: len(generated_token_ids) - len(seq)]
                return tokenizer.detokenize(generated_token_ids)

    return tokenizer.detokenize(generated_token_ids)


def fill_think_suffix(ctx: Context, model_name: str) -> None:
    if model_name.startswith("llama-3"):
        ctx.fill("\n\n")
    elif model_name.startswith("Qwen/Qwen3"):
        ctx.fill("\n\n<think></think>\n\n")
    elif model_name.startswith("deepseek-r1-distill-qwen-2"):
        ctx.fill("\n</think>\n\n")


def main() -> None:
    args = get_arguments()
    prompt = args.get(
        "prompt",
        "Generate a profile for a fictional software engineer named Alice.",
    )
    max_retries = int(args.get("max_retries", "3"))
    max_tokens = int(args.get("max_tokens", "512"))

    validator = SchemaValidator(PERSON_SCHEMA)

    model = Model.get_auto()
    model_name = model.get_name()
    eos_sequences = model.eos_tokens()
    tokenizer = model.get_tokenizer()

    escape_non_printable = model_name.startswith(
        "Qwen/Qwen3"
    ) or model_name.startswith("deepseek-r1-distill-qwen-2")

    eot_token_id = None
    for eos_seq in eos_sequences:
        if len(eos_seq) == 1:
            eot_token_id = eos_seq[0]
            break

    if eot_token_id is None:
        print(f"No single EOS token found for model: {model_name}")
        return

    if not (
        model_name.startswith("llama-3")
        or model_name.startswith("Qwen/Qwen3")
        or model_name.startswith("deepseek-r1-distill-qwen-2")
    ):
        print(
            f"JSON schema validation example is only implemented for "
            f"Llama 3, Qwen 3, and DeepSeek R1 Distill Qwen 2. Got: {model_name}"
        )
        return

    ctx = Context(model)
    ctx.fill_system(SYSTEM_PROMPT)
    ctx.fill_user(build_initial_prompt(str(prompt), PERSON_SCHEMA))
    fill_think_suffix(ctx, model_name)

    valid_result = None

    for attempt in range(1, max_retries + 1):
        print(f"--- Attempt {attempt}/{max_retries} ---")

        vocab_ids, vocab_bytes = tokenizer.get_vocabs()
        special_token_ids, special_token_bytes = tokenizer.get_special_tokens()
        constrained_sampler = ConstrainedSampler(
            vocab_ids,
            vocab_bytes,
            special_token_ids,
            special_token_bytes,
            tokenizer.get_split_regex(),
            JSON_GRAMMAR,
            eot_token_id,
            escape_non_printable,
        )

        output = generate_json(ctx, constrained_sampler, tokenizer, eos_sequences, max_tokens)
        print(f"Output: {output}")

        try:
            parsed = validator.validate(output)
            print("Schema validation passed!")
            valid_result = parsed
            break
        except Exception as e:
            error_report = str(e)
            print(f"Validation errors:\n{error_report}")
            ctx.fill_user(build_retry_prompt(error_report))

        fill_think_suffix(ctx, model_name)

    print("\n--- Result ---")
    if valid_result is not None:
        print(f"Valid JSON:\n{valid_result}")
    else:
        print(f"Failed to produce valid JSON after {max_retries} attempts.")

    set_return(valid_result or "")


if __name__ == "__main__":
    main()
