// Import Args, Result, anyhow, and block_on from the run bindings
use inferlib_run_bindings::{Args, Result, anyhow, block_on};
// Import environment functions from the bindings crate
use inferlib_environment_bindings::{get_arguments, set_return};
// Import the chat formatter from the bindings crate
use inferlib_chat_bindings::ChatFormatter;
// Import context and model from the bindings crate
use inferlib_context_bindings::{Context, Model, SamplerConfig, StopConfig};

async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    // Get the auto-selected model using the new Model library
    let model = Model::get_auto();

    // Create context using the new standalone library, passing the model reference
    let ctx = Context::new(&model);

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

    // 5. Configure sampler and stop condition using the new types
    let sampler = SamplerConfig::TopP((0.6, 0.95));
    let stop_config = StopConfig {
        max_tokens: max_num_outputs as u32,
        eos_sequences: model.eos_tokens(),
    };

    // 6. Generate using the new context library
    let final_text = ctx.generate(sampler, &stop_config);

    Ok(final_text)
}

struct __PieMain;

impl inferlib_run_bindings::Guest for __PieMain {
    fn run() -> Result<(), String> {
        let args = Args::from_vec(
            get_arguments()
                .into_iter()
                .map(std::ffi::OsString::from)
                .collect(),
        );

        let result = block_on(async { main(args).await });

        match result {
            Ok(r) => {
                // This block contains the new logic.
                use std::any::Any;
                let r_any: &dyn Any = &r;
                let output = if let Some(s) = r_any.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = r_any.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    // Fallback for all other types
                    format!("{:?}", r)
                };

                set_return(&output);
                Ok(())
            }
            Err(e) => Err(format!("{:?}", e)),
        }
    }
}

inferlib_run_bindings::export!(__PieMain with_types_in inferlib_run_bindings);
