"""
Template-driven generation with grammar-constrained decoding (Python).

Combines three layers:
1. A Lark grammar via inferlib-llguidance ensures every generated token
   produces syntactically valid JSON (no parse errors possible).
2. The inferlib-schema component validates the JSON against a product schema.
3. The inferlib-template component renders the validated data through a
   Jinja2-style template. When validation or rendering fails, the error is
   fed back and the model regenerates -- still grammar-constrained.
"""

from inference_bindings import (
    Context,
    Model,
    set_return,
)
from llguidance_bindings import ConstrainedSampler
from run_bindings import get_arguments
from schema_bindings import SchemaValidator
from template_bindings import TemplateRenderer

JSON_GRAMMAR = r"""
?start: value
?value: object
        | array
        | string
        | NUMBER             -> number
        | "true"             -> true
        | "false"            -> false
        | "null"             -> null
array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : string ":" value
string : ESCAPED_STRING
NUMBER : /-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""

TEMPLATE = r"""
========================================
  PRODUCT ANNOUNCEMENT
========================================

{{ product_name | upper }}
"{{ tagline }}"

TARGET AUDIENCE
---------------
{{ target_audience }}

OVERVIEW
--------
{{ description }}

KEY FEATURES
------------
{% for feature in features %}
  * {{ feature }}
{% endfor %}

TECHNICAL SPECIFICATIONS
------------------------
{% for spec in technical_specs %}
  * {{ spec }}
{% endfor %}

USE CASES
---------
{% for use_case in use_cases %}
  * {{ use_case }}
{% endfor %}

COMPETITIVE ADVANTAGES
----------------------
{% for advantage in competitive_advantages %}
  * {{ advantage }}
{% endfor %}

FAQ
---
{% for item in faq %}
Q: {{ item.question }}
A: {{ item.answer }}

{% endfor %}
PRICING & AVAILABILITY
----------------------
  Price: ${{ price }}
  Release Date: {{ release_date }}
{% if discount_percent %}
  Launch Discount: {{ discount_percent }}% off!
{% endif %}

========================================
"""

PRODUCT_SCHEMA = """{
    "type": "object",
    "properties": {
        "product_name": {
            "type": "string",
            "minLength": 1
        },
        "tagline": {
            "type": "string",
            "minLength": 1
        },
        "description": {
            "type": "string",
            "minLength": 1
        },
        "target_audience": {
            "type": "string",
            "minLength": 1
        },
        "features": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 5
        },
        "technical_specs": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 5
        },
        "use_cases": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 4
        },
        "competitive_advantages": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 4
        },
        "faq": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": { "type": "string" },
                    "answer":   { "type": "string" }
                },
                "required": ["question", "answer"]
            },
            "minItems": 4
        },
        "price": {
            "type": "string"
        },
        "release_date": {
            "type": "string"
        },
        "discount_percent": {
            "type": ["integer", "null"]
        }
    },
    "required": [
        "product_name", "tagline", "description", "target_audience",
        "features", "technical_specs", "use_cases", "competitive_advantages",
        "faq", "price", "release_date"
    ]
}"""

SYSTEM_PROMPT = (
    "You are a helpful assistant that generates structured product data. "
    "Output ONLY a raw JSON object with no additional text, markdown fences, or explanation. "
    "The JSON must conform to the JSON Schema provided in the user message. "
    "Be as detailed and verbose as possible: write long, richly descriptive strings for every "
    "field, populate every array with at least the required minimum number of items, and include "
    "all optional fields. "
    "If you receive validation or rendering errors, fix the JSON to address the issues "
    "and output only the corrected JSON object."
)


def build_initial_prompt(user_prompt: str, schema: str) -> str:
    return (
        f"Generate product announcement data for: {user_prompt}.\n\n"
        f"The output must conform to this JSON Schema:\n{schema}\n\n"
        "Output only the JSON object, nothing else."
    )


def build_retry_prompt(errors: str) -> str:
    return (
        f"The JSON you produced has validation/rendering errors:\n{errors}\n\n"
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
    prompt = args.get("prompt", args.get("p", "a comprehensive enterprise cloud computing platform with integrated AI, security, and analytics services"))
    max_retries = int(args.get("max-retries", args.get("r", "3")))
    max_tokens = int(args.get("max-tokens", args.get("t", "4096")))

    validator = SchemaValidator(PRODUCT_SCHEMA)
    renderer = TemplateRenderer("announcement", TEMPLATE)

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
            f"Template generation example is only implemented for "
            f"Llama 3, Qwen 3, and DeepSeek R1 Distill Qwen 2. Got: {model_name}"
        )
        return

    ctx = Context(model)
    ctx.fill_system(SYSTEM_PROMPT)
    ctx.fill_user(build_initial_prompt(str(prompt), PRODUCT_SCHEMA))
    fill_think_suffix(ctx, model_name)

    rendered_result = None

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
        print(f"Raw JSON output: {output}")

        try:
            parsed = validator.validate(output)
            print("Schema validation passed!")
            try:
                rendered = renderer.render(parsed)
                print("Template rendered successfully!")
                rendered_result = rendered
                break
            except Exception as e:
                error_msg = str(e)
                print(error_msg)
                ctx.fill_user(build_retry_prompt(error_msg))
        except Exception as e:
            error_report = str(e)
            print(f"Validation errors:\n{error_report}")
            ctx.fill_user(build_retry_prompt(error_report))

        fill_think_suffix(ctx, model_name)

    print("\n--- Result ---")
    if rendered_result is not None:
        print(rendered_result)
    else:
        print(f"Failed to produce valid renderable JSON after {max_retries} attempts.")

    set_return(rendered_result or "")


if __name__ == "__main__":
    main()
