//! Text completion with greedy (deterministic) sampling.
//!
//! Identical to `text-completion` but uses greedy decoding instead of
//! top-p sampling, guaranteeing reproducible output for the same prompt.

use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use inferlet::{Args, Result, Sampler};

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"]).unwrap_or_else(|_| {
        "Explain what makes a good unit test in three concise points.".to_string()
    });
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);
    let system_message: String = args
        .value_from_str(["-s", "--system"])
        .unwrap_or_else(|_| "You are a helpful, respectful and honest assistant.".to_string());

    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();

    ctx.fill_system(&system_message);
    ctx.fill_user(&prompt);

    let stop_cond = max_len(max_num_outputs).or(ends_with_any(model.eos_tokens()));
    let final_text = ctx.generate(Sampler::greedy(), stop_cond).await;

    println!("Output: {:?}", final_text);

    Ok(final_text)
}
