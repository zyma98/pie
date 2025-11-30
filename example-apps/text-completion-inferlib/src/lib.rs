use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use inferlet::{Args, Result, Sampler, anyhow};
use std::time::Instant;

// Import the chat formatter from the bindings crate
use inferlib_chat_bindings::ChatFormatter;

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();
    let mut ctx = model.create_context();

    // 1. Instantiate a ChatFormatter from the new WASM library
    let formatter = ChatFormatter::new(&model.get_prompt_template())
        .map_err(|e| anyhow!("Failed to create ChatFormatter: {}", e))?;

    // 2. Call the system and user methods on the new library
    formatter.add_system("You are a helpful, respectful and honest assistant.");
    formatter.add_user(&prompt);

    // 3. Generate the rendered string with the new library
    let rendered_prompt = formatter.render(true, true);

    // 4. Feed the rendered string into ctx.fill()
    ctx.fill(&rendered_prompt);

    let sampler = Sampler::top_p(0.6, 0.95);
    let stop_cond = max_len(max_num_outputs).or(ends_with_any(model.eos_tokens()));

    let final_text = ctx.generate(sampler, stop_cond).await;

    let token_ids = tokenizer.tokenize(&final_text);
    println!(
        "Output: {:?} (total elapsed: {:?})",
        final_text,
        start.elapsed()
    );

    // Compute per-token latency, avoiding division by zero.
    if !token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (token_ids.len() as u32)
        );
    }

    Ok(final_text)
}
