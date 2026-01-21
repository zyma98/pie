use inferlib_chat_bindings::ChatFormatter;
use inferlib_context_bindings::{Context, Model, SamplerConfig, StopConfig};
use inferlib_run_bindings::{Args, Result, anyhow};
use std::time::Instant;

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    let start = Instant::now();

    let model = Model::get_auto();
    println!("Model: {}", model.get_name());

    let tokenizer = model.get_tokenizer();
    let ctx = Context::new(&model);

    // let formatter = ChatFormatter::new(&model.get_prompt_template())
    //     .map_err(|e| anyhow!("Failed to create ChatFormatter: {}", e))?;

    // formatter.add_system("You are a helpful, respectful and honest assistant.");
    // formatter.add_user(&prompt);

    // let rendered_prompt = formatter.render(true, true);
    // ctx.fill(&rendered_prompt);

    // let sampler = SamplerConfig::TopP((0.6, 0.95));
    // let stop_config = StopConfig {
    //     max_tokens: max_num_outputs as u32,
    //     eos_sequences: model.eos_tokens(),
    // };

    // let final_text = ctx.generate(sampler, &stop_config);

    // let token_ids = tokenizer.tokenize(&final_text);
    // println!(
    //     "Output: {:?} (total elapsed: {:?})",
    //     final_text,
    //     start.elapsed()
    // );

    // // Compute per-token latency, avoiding division by zero.
    // if !token_ids.is_empty() {
    //     println!(
    //         "Per token latency: {:?}",
    //         start.elapsed() / (token_ids.len() as u32)
    //     );
    // }

    // Ok(final_text)

    Ok("Hello, world!".to_string())
}
