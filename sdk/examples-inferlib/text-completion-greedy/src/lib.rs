//! Text completion with greedy (deterministic) sampling using inferlib.
//!
//! Identical to `text-completion-inferlib` but uses greedy decoding instead of
//! top-p sampling, guaranteeing reproducible output for the same prompt.

use inferlib_inference_bindings::{Context, Model, SamplerConfig, StopConfig};
use inferlib_run_bindings::{Args, Result};

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"]).unwrap_or_else(|_| {
        "Explain what makes a good unit test in three concise points.".to_string()
    });
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);
    let system_message: String = args
        .value_from_str(["-s", "--system"])
        .unwrap_or_else(|_| "You are a helpful, respectful and honest assistant.".to_string());

    let model = Model::get_auto();
    let ctx = Context::new(&model);

    ctx.fill_system(&system_message);
    ctx.fill_user(&prompt);

    let stop_config = StopConfig {
        max_tokens: max_num_outputs as u32,
        eos_sequences: model.eos_tokens(),
    };

    let final_text = ctx.generate(SamplerConfig::Greedy, &stop_config);

    println!("Output: {:?}", final_text);

    Ok(final_text)
}
