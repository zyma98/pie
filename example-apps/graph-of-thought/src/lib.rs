use futures::future::join_all;
use futures::{StreamExt, stream::FuturesUnordered}; // StreamExt is needed for `next()`

use symphony::{Context, Model};

const PROPOSE_PROMPT_TEMPLATE: &str = "Could you suggest a method or approach to solve the following question? Please provide a high-level plan without doing the actual calculation. Keep it concise, around 80 words. Question: {}";
const AGGREGATE_PROMPT: &str = "Please compare the following solution with the one you just provided and aggregate their ideas into a single, improved solution:\n";
// Other constants (ASSISTANT_PREFIX, STOP_TOKEN, MAX_TOKENS) remain unchanged
const ASSISTANT_PREFIX: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";
const STOP_TOKEN: &str = "<|eot_id|>";
const MAX_TOKENS: usize = 256;

async fn async_aggregation(mut init_ctx: Context, question: &str) -> Vec<String> {
    let propose_prompt = format!("{} {}", PROPOSE_PROMPT_TEMPLATE, question);

    init_ctx.fill(&propose_prompt);
    init_ctx.fill(ASSISTANT_PREFIX);

    // Gen
    let mut aggregate_level1 = [16, 128, 32, 256, 128, 128, 16, 32]
        .into_iter()
        .map(|max_tokens| {
            let mut ctx = init_ctx.fork();
            async move { (ctx.generate_until(STOP_TOKEN, max_tokens).await, ctx) }
        })
        .collect::<FuturesUnordered<_>>();

    let mut aggregate_level2 = FuturesUnordered::<_>::new();
    let mut waiting_pair = None;

    while let Some((output, mut ctx)) = aggregate_level1.next().await {
        if waiting_pair.is_none() {
            waiting_pair = Some(output);
        } else {
            let pair = waiting_pair.take().unwrap();
            ctx.fill(AGGREGATE_PROMPT);
            ctx.fill(&pair);
            ctx.fill(ASSISTANT_PREFIX);
            aggregate_level2.push(async move { (ctx.generate_until(STOP_TOKEN, 32).await, ctx) });
        }
    }

    let mut aggregate_level3 = Vec::new();
    let mut waiting_pair = None;

    while let Some((output, mut ctx)) = aggregate_level2.next().await {
        if waiting_pair.is_none() {
            waiting_pair = Some(output);
        } else {
            let pair = waiting_pair.take().unwrap();
            ctx.fill(AGGREGATE_PROMPT);
            ctx.fill(&pair);
            ctx.fill(ASSISTANT_PREFIX);
            aggregate_level3.push(async move { ctx.generate_until(STOP_TOKEN, 32).await });
        }
    }

    join_all(aggregate_level3).await
}

#[symphony::main]
async fn main() -> Result<(), String> {
    // Initialize the Symphony model and a common context.
    let available_models = symphony::available_models();

    let model = Model::new(available_models.first().unwrap()).unwrap();
    let mut ctx = model.create_context();
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");

    let question = "What is the sum of 123456789 and 987654321?";

    //tree_search(ctx, question, num_branches).await;
    //tree_search_branch_parallel(ctx, question, num_branches).await;\
    async_aggregation(ctx, question).await;
    Ok(())
}
