//! Demonstrates Tree-of-Thought (ToT) for multi-branch reasoning.
//!
//! This example performs a 3-level tree search (Propose, Execute, Reflect) where
//! each level spawns multiple branches. All branches are explored concurrently,
//! leveraging KV cache sharing from common prefixes.

use futures::future;
use inferlib_inference_bindings::{Context, GenerateFuture, Model, SamplerConfig, StopConfig};
use inferlib_run_bindings::{Args, Result};
use std::time::Instant;
use wstd::runtime::AsyncPollable;

const HELP: &str = "\
Usage: tree_of_thought [OPTIONS]

A program to perform a 3-level tree of thought (Propose, Execute, Reflect) search.

Options:
  -q, --question <TEXT>        The question to solve [default: Calculate (42 + 3) * 5 / 15.]
  -b, --num-branches <INT>     Number of branches at each level of the tree [default: 2]
  -t, --max-tokens <INT>       Max new tokens to generate at each step [default: 512]
  -h, --help                   Prints this help message";

const PROPOSE_PROMPT_TEMPLATE: &str = "\
Please generate a high-level plan for solving the following question. \
First, just state the method you will use. Do not do the actual calculation. \
Keep your response concise and within 80 words. Question: ";

const EXECUTE_PROMPT: &str = "\
The plan looks good! Now, use real numbers and do the calculation. \
Please solve the question step-by-step according to the plan. \
Give me the final answer. Make your response short.";

const REFLECT_PROMPT: &str = "\
Okay. Now, evaluate your own solution and give it a score on a scale of 1 to 5. \
Please rigorously check the correctness of the calculations and the final answer.";

fn make_stop_config(max_tokens: u32, eos: &[Vec<u32>]) -> StopConfig {
    StopConfig {
        max_tokens,
        eos_sequences: eos.to_vec(),
    }
}

async fn poll_generate(gen_future: &GenerateFuture) -> String {
    loop {
        let pollable = gen_future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        if let Some(result) = gen_future.get() {
            return result;
        }
    }
}

async fn poll_flush(ctx: &Context) {
    if let Some(flush_future) = ctx.flush_async() {
        let pollable = flush_future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
    }
}

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let question: String = args
        .value_from_str(["-q", "--question"])
        .unwrap_or_else(|_| "Calculate (42 + 3) * 5 / 15.".to_string());

    let num_branches: usize = args.value_from_str(["-b", "--num-branches"]).unwrap_or(2);
    let max_tokens_per_step: usize = args.value_from_str(["-t", "--max-tokens"]).unwrap_or(512);

    let total_leaves = num_branches.pow(3);
    println!(
        "--- Starting Tree of Thought (Branches={}, Leaves={}, MaxTokens/Step={}) ---",
        num_branches, total_leaves, max_tokens_per_step
    );
    let start = Instant::now();

    let model = Model::get_auto();
    let eos_tokens = model.eos_tokens();
    let ctx_root = Context::new(&model);

    ctx_root.fill_system(
        "You are a helpful, respectful, and honest assistant that excels at \
        mathematical reasoning. Please follow the user's instructions precisely.",
    );
    poll_flush(&ctx_root).await;

    // Build and execute tree in parallel
    let level1_futures = (0..num_branches).map(|_| {
        let propose_ctx = ctx_root.fork();
        let question_ = question.clone();
        let eos_tokens = eos_tokens.clone();
        async move {
            // Level 1: Propose Plan
            let propose_prompt = format!("{}{}", PROPOSE_PROMPT_TEMPLATE, question_);
            propose_ctx.fill_user(&propose_prompt);

            let stop_config = make_stop_config(max_tokens_per_step as u32, &eos_tokens);
            let gen_future =
                propose_ctx.generate_async(SamplerConfig::TopP((0.6, 0.95)), &stop_config);
            poll_generate(&gen_future).await;

            // Level 2: Execute Plan
            propose_ctx.fill_user(EXECUTE_PROMPT);
            poll_flush(&propose_ctx).await;

            let level2_futures = (0..num_branches).map(|_| {
                let execute_ctx = propose_ctx.fork();
                let eos_tokens = eos_tokens.clone();
                async move {
                    let stop_config = make_stop_config(max_tokens_per_step as u32, &eos_tokens);
                    let gen_future =
                        execute_ctx.generate_async(SamplerConfig::TopP((0.6, 0.95)), &stop_config);
                    poll_generate(&gen_future).await;

                    // Level 3: Reflect on Solution
                    execute_ctx.fill_user(REFLECT_PROMPT);
                    poll_flush(&execute_ctx).await;

                    let level3_futures = (0..num_branches).map(|_| {
                        let reflect_ctx = execute_ctx.fork();
                        let eos_tokens = eos_tokens.clone();
                        async move {
                            let stop_config =
                                make_stop_config(max_tokens_per_step as u32, &eos_tokens);
                            let gen_future = reflect_ctx
                                .generate_async(SamplerConfig::TopP((0.6, 0.95)), &stop_config);
                            poll_generate(&gen_future).await;
                            reflect_ctx
                        }
                    });
                    future::join_all(level3_futures).await
                }
            });
            future::join_all(level2_futures).await
        }
    });

    // Await all top-level branches, which in turn await their sub-branches
    let nested_results: Vec<Vec<Vec<Context>>> = future::join_all(level1_futures).await;

    // Flatten the results to get a single list of all leaf contexts
    let final_ctxs: Vec<Context> = nested_results.into_iter().flatten().flatten().collect();

    println!(
        "\n--- All {} leaf nodes generated in {:?} ---\n",
        final_ctxs.len(),
        start.elapsed()
    );

    if let Some(last_ctx) = final_ctxs.last() {
        println!(
            "Sample Result (Leaf #{}):\n{}\n",
            final_ctxs.len(),
            last_ctx.get_text()
        );
    } else {
        println!("No results were generated.");
    }

    Ok(())
}
