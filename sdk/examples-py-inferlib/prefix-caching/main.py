"""
Prefix caching example using inferlib (Python).

Caches a long system prompt's KV state and reuses it for subsequent requests,
avoiding redundant prefill computation.
"""

import json

from inference_bindings import (
    ChatFormatter,
    Context,
    Model,
    Queue,
    SamplerConfig_Greedy,
    StopConfig,
    set_return,
    store_get,
    store_set,
)
from run_bindings import get_arguments

PREFIX_TO_CACHE = r"""# **Core Identity: The Digital Teacher**

You are an advanced AI assistant, but your core persona is that of a kind, patient, and incredibly knowledgeable Korean elementary school teacher. Your primary goal is not just to answer questions, but to educate, enlighten, and encourage curiosity in a supportive and structured manner, as if you are leading a classroom of bright young students.

## **Persona Directive: The Korean Elementary School Teacher**

Your entire response, regardless of the topic, must be delivered in this specific persona.

* **Tone & Style:**
    * **Warm & Encouraging:** Use positive and uplifting language.
    * **Patient & Clear:** Explain complex topics using simple, step-by-step logic.
    * **Use Analogies:** Relate complex ideas to simple, everyday concepts.
    * **Structured like a Lesson:** Begin with a friendly opening, present the main lesson in a clear way, and conclude with an encouraging closing remark.

* **Behavioral Guidelines:**
    * Address the user respectfully, as you would a student.
    * Never be condescending or impatient.
    * Celebrate curiosity and praise the user for asking good questions.

## **Core Principles of Your Responses**

1.  **Clarity:** Explain concepts as if teaching for the first time.
2.  **Accuracy:** Your facts must be correct.
3.  **Structure:** Organize your answers like a good lesson plan.
4.  **Safety & Ethics:** Politely decline harmful or unethical requests.
5.  **Conciseness:** Stay on topic.

You are now ready to help your student.
"""

CACHE_FLAG_KEY = "prefix_loaded_v1"
CACHE_EXPORT_NAME = "my_system_prefix_v1"
CACHE_STATE_KEY = "my_system_prefix_state_v1"


def main() -> None:
    args = get_arguments()
    prompt = args.get("prompt", "What is the capital of Washington State?")
    max_tokens = int(args.get("max_tokens", "128"))
    invalidate_cache = bool(args.get("invalidate_cache", False))

    model = Model.get_auto()
    queue = Queue.from_model_name(model.get_name())
    tokenizer = model.get_tokenizer()

    if invalidate_cache and store_get(CACHE_FLAG_KEY) == "true":
        queue.release_exported_kv_pages(CACHE_EXPORT_NAME)
        store_set(CACHE_FLAG_KEY, "false")

    if store_get(CACHE_FLAG_KEY) == "true":
        print("Cache HIT. Loading prefix from KV store.")

        imported_page_ids = queue.import_kv_pages(CACHE_EXPORT_NAME)
        state_json = store_get(CACHE_STATE_KEY)
        if state_json is None:
            print("Cache Inconsistency: State missing")
            return
        state = json.loads(state_json)

        ctx = Context.from_imported_state(
            model,
            imported_page_ids,
            state["token_ids"],
            state["kv_page_last_len"],
        )
    else:
        print("Cache MISS. Computing and caching prefix.")

        formatter = ChatFormatter(model.get_prompt_template())
        formatter.add_system(PREFIX_TO_CACHE)
        system_prompt = formatter.render(False, True)

        prefill_ctx = Context(model)
        prefill_ctx.fill(system_prompt)
        prefill_ctx.flush()

        state_to_cache = {
            "token_ids": prefill_ctx.get_token_ids(),
            "kv_page_last_len": prefill_ctx.get_kv_page_last_len(),
        }

        queue.export_kv_pages(prefill_ctx.get_kv_page_ptrs(), CACHE_EXPORT_NAME)

        store_set(CACHE_STATE_KEY, json.dumps(state_to_cache))
        store_set(CACHE_FLAG_KEY, "true")

        ctx = prefill_ctx

    ctx.fill_user(prompt)

    stop_config = StopConfig(
        max_tokens=max_tokens, eos_sequences=model.eos_tokens()
    )

    text = ctx.generate(SamplerConfig_Greedy(), stop_config)
    token_ids = tokenizer.tokenize(text)
    print(f"Output: {text!r}")

    if token_ids:
        print(f"Tokens generated: {len(token_ids)}")

    set_return(text)


if __name__ == "__main__":
    main()
