//! Demonstrates Skeleton-of-Thought (SoT) for parallel elaboration.
//!
//! This example first generates a high-level plan (skeleton) with key points,
//! then elaborates on each point concurrently. This approach can reduce latency
//! by parallelizing the detailed generation phase.

use futures::future;
use inferlib_inference_bindings::{Context, GenerateFuture, Model, SamplerConfig, StopConfig};
use inferlib_run_bindings::{Args, Result};
use std::time::Instant;
use wstd::runtime::AsyncPollable;

const HELP: &str = "\
Usage: skeleton-of-thought [OPTIONS]

A program that first generates a plan (a list of key points) for a question,
and then elaborates on each point concurrently.

Options:
  -q, --question <TEXT>        The question to answer [default: What are the defining characteristics of Rome?]
  -p, --num-points <POINTS>    Sets the maximum number of key points to generate in the plan [default: 3]
  -t, --plan-tokens <TOKENS>   Sets the max tokens for the planning generation [default: 256]
  -e, --elab-tokens <TOKENS>   Sets the max tokens for each elaboration generation [default: 256]
  -h, --help                   Prints this help message";

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

/// Generates a high-level plan and elaborates on each point in parallel.
async fn plan_and_generate_parallel(
    ctx: Context,
    question: &str,
    max_points: usize,
    plan_max_tokens: usize,
    elab_max_tokens: usize,
    eos_tokens: &Vec<Vec<u32>>,
) -> Vec<String> {
    // 1. Fork a context for generating the plan.
    let plan_ctx = ctx.fork();
    let plan_prompt = format!(
        "Generate up to {} key points that outline the answer to the following question: {}. \
        Each point must be enclosed between the <point> and </point> tags.",
        max_points, question
    );
    plan_ctx.fill_user(&plan_prompt);

    let stop_config = make_stop_config(plan_max_tokens as u32, eos_tokens);
    let gen_future = plan_ctx.generate_async(SamplerConfig::TopP((0.6, 0.95)), &stop_config);
    let output = poll_generate(&gen_future).await;

    // 2. Robustly parse points from the output.
    let points: Vec<String> = output
        .split("<point>")
        .skip(1)
        .filter_map(|s| s.split("</point>").next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if points.is_empty() {
        return Vec::new();
    }

    // 3. Fork from the original base context for a clean state for each elaboration.
    let leaf_futures = points
        .into_iter()
        .map(|point| {
            let elab_ctx = ctx.fork();
            let complete_prompt = format!(
                "Elaborate on the following point: {}. \
                Your response should be complete and only concerned with this point.",
                point
            );
            elab_ctx.fill_user(&complete_prompt);

            let eos_tokens = eos_tokens.clone();
            async move {
                let stop_config = make_stop_config(elab_max_tokens as u32, &eos_tokens);
                let gen_future =
                    elab_ctx.generate_async(SamplerConfig::TopP((0.6, 0.95)), &stop_config);
                poll_generate(&gen_future).await
            }
        })
        .collect::<Vec<_>>();

    future::join_all(leaf_futures).await
}

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let question: String = args
        .value_from_str(["-q", "--question"])
        .unwrap_or_else(|_| "What are the defining characteristics of Rome?".to_string());

    let num_points: usize = args.value_from_str(["-p", "--num-points"]).unwrap_or(3);
    let plan_max_tokens: usize = args.value_from_str(["-t", "--plan-tokens"]).unwrap_or(256);
    let elab_max_tokens: usize = args.value_from_str(["-e", "--elab-tokens"]).unwrap_or(256);

    let start = Instant::now();

    let model = Model::get_auto();
    let eos_tokens = model.eos_tokens();
    let ctx = Context::new(&model);

    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    poll_flush(&ctx).await;

    println!(
        "--- Starting plan and generate (plan: {} points, {} tokens; elab: {} tokens) ---",
        num_points, plan_max_tokens, elab_max_tokens
    );

    let elaborations = plan_and_generate_parallel(
        ctx,
        &question,
        num_points,
        plan_max_tokens,
        elab_max_tokens,
        &eos_tokens,
    )
    .await;

    println!("\n--- Completed in {:?} ---\n", start.elapsed());

    if elaborations.is_empty() {
        println!("No points were generated or elaborated upon.");
    } else {
        for (i, elaboration) in elaborations.iter().enumerate() {
            println!("Elaboration {}:\n{}\n", i + 1, elaboration);
        }
    }

    Ok(())
}
