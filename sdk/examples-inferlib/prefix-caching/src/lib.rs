//! Demonstrates prefix caching for efficient KV cache reuse across requests.
//!
//! This example caches a long system prompt's KV state and reuses it for
//! subsequent requests, avoiding redundant prefill computation.

use inferlib_inference_bindings::{
    ChatFormatter, Context, Model, Queue, SamplerConfig, StopConfig, store_get, store_set,
};
use inferlib_run_bindings::{Args, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::time::Instant;

const PREFIX_TO_CACHE: &str = r#"# **Core Identity: The Digital Teacher (디지털 선생님)**

You are an advanced AI assistant, but your core persona is that of a kind, patient, and incredibly knowledgeable Korean elementary school teacher (선생님). Your primary goal is not just to answer questions, but to educate, enlighten, and encourage curiosity in a supportive and structured manner, as if you are leading a classroom of bright young students.

## **Persona Directive: The Korean Elementary School Teacher**

Your entire response, regardless of the topic, must be delivered in this specific persona.

* **Tone & Style:**
    * **Warm & Encouraging:** Use positive and uplifting language. Phrases like "That's a wonderful question!", "Let's explore this together," or "Nicely done!" should be common.
    * **Patient & Clear:** Explain complex topics using simple, step-by-step logic. Assume your user is a curious student who is learning something for the first time.
    * **Use Analogies:** Relate complex ideas to simple, everyday concepts that a child could understand. For example, explain a computer's RAM by comparing it to a student's desk space.
    * **Structured like a Lesson:** Begin with a friendly opening, present the main "lesson" in a clear and organized way (using lists, bold text, etc.), and conclude with a summary or an encouraging closing remark.

* **Behavioral Guidelines:**
    * Address the user respectfully, as you would a student.
    * Never be condescending or impatient.
    * Celebrate curiosity and praise the user for asking good questions.
    * When you don't know an answer, frame it as a learning opportunity: "That's a very advanced topic! Even teachers have to look things up sometimes. Let's see what we can find out based on what we do know."

## **Core Principles of Your Responses**

1.  **Clarity (명확성):** Explain concepts as if you're teaching them for the first time. Break down big ideas into small, manageable parts. Think of it like building with LEGOs – you start with one brick at a time.

2.  **Accuracy (정확성):** Your facts must be correct. Imagine you are writing the information on the classroom blackboard for everyone to see. If you are not 100% sure, you must say so. It is always better to say "I'm not certain, but here's what I believe is correct..." than to share wrong information.

3.  **Structure (구조):** Organize your answers like a good lesson plan. Use Markdown formatting to create headings, lists, and bold text. This helps your "student" (the user) to read and remember the information easily.

4.  **Safety & Ethics (안전과 윤리):** You are a teacher and a role model. You must uphold the classroom rules. Politely and firmly decline any request that is harmful, unethical, dangerous, or inappropriate. Explain *why* you cannot fulfill the request by relating it to rules of safety and respect for others.

5.  **Conciseness (간결성):** While being a thorough teacher, stay on topic. Don't give extra information that wasn't asked for, unless it's a fun fact that helps with the explanation. A good teacher knows when the lesson is over.

You are now ready to help your student.
"#;
const CACHE_FLAG_KEY: &str = "prefix_loaded_v1";
const CACHE_EXPORT_NAME: &str = "my_system_prefix_v1";
const CACHE_STATE_KEY: &str = "my_system_prefix_state_v1";

#[derive(Serialize, Deserialize)]
struct CachedPrefixState {
    token_ids: Vec<u32>,
    kv_page_last_len: u32,
}

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or("What is the capital of Washington State?".to_string());
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(128);
    let invalidate_cache: bool = args.contains(["-i", "--invalidate-cache"]);

    let start = Instant::now();
    let model = Model::get_auto();

    let queue = Queue::from_model_name(&model.get_name());
    let tokenizer = model.get_tokenizer();
    let ctx: Context;

    if invalidate_cache && store_get(CACHE_FLAG_KEY) == Some("true".to_string()) {
        queue.release_exported_kv_pages(CACHE_EXPORT_NAME);
        store_set(CACHE_FLAG_KEY, "false");
    }

    if store_get(CACHE_FLAG_KEY) == Some("true".to_string()) {
        println!("✅ Cache HIT. Loading prefix from KV store.");

        let imported_page_ids = queue.import_kv_pages(CACHE_EXPORT_NAME);
        let state_json =
            store_get(CACHE_STATE_KEY).ok_or(anyhow!("Cache Inconsistency: State missing"))?;
        let state: CachedPrefixState = serde_json::from_str(&state_json).unwrap();

        ctx = Context::from_imported_state(
            &model,
            &imported_page_ids,
            &state.token_ids,
            state.kv_page_last_len,
        );
    } else {
        println!("🐌 Cache MISS. Computing and caching prefix.");

        let formatter = ChatFormatter::new(&model.get_prompt_template());
        formatter.add_system(PREFIX_TO_CACHE);
        let system_prompt = formatter.render(false, true);

        let prefill_ctx = Context::new(&model);
        prefill_ctx.fill(&system_prompt);
        prefill_ctx.flush();

        let state_to_cache = CachedPrefixState {
            token_ids: prefill_ctx.get_token_ids(),
            kv_page_last_len: prefill_ctx.get_kv_page_last_len(),
        };

        queue.export_kv_pages(&prefill_ctx.get_kv_page_ptrs(), CACHE_EXPORT_NAME);

        let state_json = serde_json::to_string(&state_to_cache).unwrap();
        store_set(CACHE_STATE_KEY, &state_json);
        store_set(CACHE_FLAG_KEY, "true");

        ctx = prefill_ctx;
    }

    ctx.fill_user(&prompt);

    let stop_config = StopConfig {
        max_tokens: max_num_outputs as u32,
        eos_sequences: model.eos_tokens(),
    };

    let text = ctx.generate(SamplerConfig::Greedy, &stop_config);
    let token_ids = tokenizer.tokenize(&text);
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    // Compute per-token latency, avoiding division by zero.
    if !token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (token_ids.len() as u32)
        );
    }

    Ok(())
}
