use inferlib3_inference_bindings::{ChatFormatter, Context, Model, SamplerConfig, StopConfig};
use inferlib_old_run_bindings::{Args, Result, anyhow};

#[inferlib3_macros::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    let model = Model::get_auto();

    let formatter = ChatFormatter::new(&model.get_prompt_template())
        .map_err(|e| anyhow!("Failed to create ChatFormatter: {}", e))?;

    formatter.add_system("You are a helpful, respectful and honest assistant.");
    formatter.add_user(&prompt);

    let rendered_prompt = formatter.render(true, true);

    let ctx = Context::new(&model);
    ctx.fill(&rendered_prompt);

    let sampler = SamplerConfig::TopP((0.6, 0.95));
    let stop_config = StopConfig {
        max_tokens: max_num_outputs as u32,
        eos_sequences: model.eos_tokens(),
    };

    let final_text = ctx.generate(sampler, &stop_config);

    Ok(final_text)
}
