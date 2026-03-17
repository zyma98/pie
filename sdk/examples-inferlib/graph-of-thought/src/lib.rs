//! Demonstrates Graph-of-Thought (GoT) for hierarchical aggregation.
//!
//! This example generates multiple initial proposals concurrently, then
//! progressively aggregates them in pairs across multiple levels. The streaming
//! nature allows aggregation to begin as soon as pairs of proposals are ready,
//! maximizing parallelism.

use futures::stream::FuturesUnordered;
use futures::{StreamExt, future};
use inferlib_inference_bindings::{Context, GenerateFuture, Model, SamplerConfig, StopConfig};
use inferlib_run_bindings::{Args, Result, anyhow};
use std::time::Instant;
use wstd::runtime::AsyncPollable;

const HELP: &str = "\
Usage: graph-of-thought [OPTIONS]

A program to test hierarchical aggregation by generating and combining multiple proposals for a given question.

Options:
  --question <QUESTION>          The question to process [default: Calculate (42 + 3) * 5 / 15.]
  --proposal-tokens <TOKENS>     Comma-separated list of max tokens for initial proposals
                                 [default: 256,256,256,256,256,256,256,256]
  --aggregation-tokens <TOKENS>  Max tokens for each aggregation step [default: 256]
  -h, --help                     Prints this help message";

const SYSTEM_PROMPT: &str = "You are a helpful, respectful and honest assistant.";

const PROPOSAL_PROMPT_TEMPLATE: &str = "\
Could you suggest a method or approach to solve the following question? \
Please provide a high-level plan without doing the actual calculation. \
Keep it concise, around 80 words. Question: {}";

const AGGREGATE_PROMPT: &str = "\
Please compare the following solution with the one you just provided \
and aggregate their ideas into a single, improved solution:\n";

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

/// Main logic for running the hierarchical aggregation workflow.
async fn run_hierarchical_aggregation(
    base_context: Context,
    question: &str,
    proposal_tokens: Vec<usize>,
    aggregation_tokens: usize,
    eos_tokens: &Vec<Vec<u32>>,
) -> Vec<String> {
    // --- Stage 1: Generate Initial Proposals ---
    let propose_prompt = PROPOSAL_PROMPT_TEMPLATE.replace("{}", question);
    base_context.fill_user(&propose_prompt);
    poll_flush(&base_context).await;

    let mut proposal_tasks = proposal_tokens
        .into_iter()
        .map(|max_tokens| {
            let ctx = base_context.fork();
            let eos_tokens = eos_tokens.clone();
            async move {
                let stop_config = make_stop_config(max_tokens as u32, &eos_tokens);
                let gen_future = ctx.generate_async(SamplerConfig::TopP((0.6, 0.95)), &stop_config);
                let proposal_text = poll_generate(&gen_future).await;
                (proposal_text, ctx)
            }
        })
        .collect::<FuturesUnordered<_>>();

    // --- Stage 2: First-Level Aggregation (Pairing Proposals) ---
    let mut first_aggregation_tasks = FuturesUnordered::new();
    let mut pending_proposal: Option<(String, Context)> = None;

    while let Some((proposal_text, proposal_ctx)) = proposal_tasks.next().await {
        if pending_proposal.is_none() {
            pending_proposal = Some((proposal_text, proposal_ctx));
        } else {
            let (previous_proposal_text, _) = pending_proposal.take().unwrap();
            let aggregation_prompt = format!("{}{}", AGGREGATE_PROMPT, previous_proposal_text);
            proposal_ctx.fill_user(&aggregation_prompt);

            let eos_tokens = eos_tokens.clone();
            first_aggregation_tasks.push(async move {
                let stop_config = make_stop_config(aggregation_tokens as u32, &eos_tokens);
                let gen_future =
                    proposal_ctx.generate_async(SamplerConfig::TopP((0.6, 0.95)), &stop_config);
                let aggregation_text = poll_generate(&gen_future).await;
                (aggregation_text, proposal_ctx)
            });
        }
    }

    // --- Stage 3: Second-Level Aggregation (Pairing Aggregations) ---
    let mut second_aggregation_tasks = Vec::new();
    let mut pending_aggregation: Option<(String, Context)> = None;

    while let Some((aggregation_text, aggregation_ctx)) = first_aggregation_tasks.next().await {
        if pending_aggregation.is_none() {
            pending_aggregation = Some((aggregation_text, aggregation_ctx));
        } else {
            let (previous_aggregation_text, _) = pending_aggregation.take().unwrap();
            let final_prompt = format!("{}{}", AGGREGATE_PROMPT, previous_aggregation_text);
            aggregation_ctx.fill_user(&final_prompt);

            let eos_tokens = eos_tokens.clone();
            second_aggregation_tasks.push(async move {
                let stop_config = make_stop_config(aggregation_tokens as u32, &eos_tokens);
                let gen_future =
                    aggregation_ctx.generate_async(SamplerConfig::TopP((0.6, 0.95)), &stop_config);
                poll_generate(&gen_future).await
            });
        }
    }

    // --- Stage 4: Collect Final Results ---
    future::join_all(second_aggregation_tasks).await
}

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let question: String = args
        .value_from_str("--question")
        .unwrap_or_else(|_| "Calculate (42 + 3) * 5 / 15.".to_string());

    let proposal_tokens_str: String = args
        .value_from_str("--proposal-tokens")
        .unwrap_or_else(|_| "256,256,256,256,256,256,256,256".to_string());

    let proposal_tokens: Vec<usize> = proposal_tokens_str
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("Failed to parse proposal tokens: {}", e))?;

    let aggregation_tokens: usize = args.value_from_str("--aggregation-tokens").unwrap_or(256);

    let start = Instant::now();
    println!(
        "--- Starting hierarchical aggregation for question: \"{}\" ---",
        question
    );
    println!(
        "Proposal tokens: {:?}, Aggregation tokens: {}",
        proposal_tokens, aggregation_tokens
    );

    let model = Model::get_auto();
    let eos_tokens = model.eos_tokens();
    let ctx_root = Context::new(&model);

    ctx_root.fill_system(SYSTEM_PROMPT);
    poll_flush(&ctx_root).await;

    let final_solutions = run_hierarchical_aggregation(
        ctx_root,
        &question,
        proposal_tokens,
        aggregation_tokens,
        &eos_tokens,
    )
    .await;

    println!("\n--- Aggregation complete in {:?} ---\n", start.elapsed());

    for (i, solution) in final_solutions.iter().enumerate() {
        println!("Final aggregated solution #{}:\n{}\n", i + 1, solution);
    }

    Ok(())
}
