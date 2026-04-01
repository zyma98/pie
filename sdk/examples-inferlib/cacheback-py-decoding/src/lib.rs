//! Demonstrates speculative decoding with a cache-based drafter (Cacheback Decoding).
//!
//! This variant uses the `inferlib-cacheback-py` Python library component for the
//! two-level LRU cache table, while keeping the Trie-based draft organization
//! logic locally.  The main model verifies speculated tokens via
//! `Context::verify_draft` in a single batched forward pass with tree attention.

use inferlib_cacheback_py_bindings::CacheTable;
use inferlib_inference_bindings::{Context, Model};
use inferlib_run_bindings::{Args, Result};
use std::time::Instant;

const HELP: &str = "\
Usage: cacheback-py-decoding [OPTIONS]

Demonstrates cacheback decoding with a Python cache-table component.

Options:
  -p, --prompt <STRING>    The prompt to send to the model
                           [default: Keep printing 'hello, world!' 100 times.]
  -n, --max-tokens <INT>   The maximum number of new tokens to generate [default: 256]
  -h, --help               Print help information";

const LEADER_LEN: usize = 1;
const FOLLOWER_LEN: usize = 2;
const LEADER_CAPACITY: u32 = 256;
const FOLLOWER_CAPACITY: u32 = 4;

struct TrieNode {
    children: Vec<TrieNode>,
    token: u32,
    position: u32,
}

impl TrieNode {
    fn new(position: u32, token: u32) -> Self {
        Self {
            children: Vec::new(),
            token,
            position,
        }
    }
}

/// A Trie forest for organizing speculative token drafts into a tree structure.
struct TrieForest {
    roots: Vec<TrieNode>,
    root_position: u32,
}

impl TrieForest {
    fn new(root_position: u32) -> Self {
        Self {
            roots: Vec::new(),
            root_position,
        }
    }

    fn insert(&mut self, tokens: &[u32], positions: &[u32]) {
        if tokens.is_empty() || positions.is_empty() {
            return;
        }

        if positions[0] != self.root_position {
            return;
        }

        let mut candidate_nodes = &mut self.roots;

        for (&token, &position) in tokens.iter().zip(positions.iter()) {
            let candidate_node = candidate_nodes
                .iter()
                .find(|node| node.position == position && node.token == token);

            if candidate_node.is_none() {
                candidate_nodes.push(TrieNode::new(position, token));
            }

            let next_node = candidate_nodes
                .iter_mut()
                .find(|node| node.position == position && node.token == token)
                .unwrap();

            candidate_nodes = &mut next_node.children;
        }
    }

    fn linearize(&self) -> (Vec<u32>, Vec<u32>) {
        fn dfs(node: &TrieNode, tokens: &mut Vec<u32>, positions: &mut Vec<u32>) {
            tokens.push(node.token);
            positions.push(node.position);
            for child in node.children.iter() {
                dfs(child, tokens, positions);
            }
        }

        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        for node in self.roots.iter() {
            dfs(node, &mut tokens, &mut positions);
        }
        (tokens, positions)
    }
}

/// Updates the cache table with new context tokens, bridging across calls
/// via `prev_window`.
fn update_cache(cache_table: &CacheTable, prev_window: &mut Vec<u32>, context: &[u32]) {
    let window_len = LEADER_LEN + FOLLOWER_LEN - 1;
    let mut full = Vec::with_capacity(prev_window.len() + context.len());
    full.extend_from_slice(prev_window);
    full.extend_from_slice(context);
    cache_table.update_cache(&full);
    *prev_window = full[full.len() - window_len..].to_vec();
}

/// Queries the cache table and organizes drafts into a linearized Trie.
fn draft(cache_table: &CacheTable, prev_window: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let positions: Vec<u32> = (1..=FOLLOWER_LEN as u32).collect();
    let mut trie = TrieForest::new(1);

    let key = &prev_window[prev_window.len() - LEADER_LEN..];
    let drafts = cache_table.get_draft_tokens(key);
    for d in &drafts {
        trie.insert(d, &positions);
    }

    trie.linearize()
}

#[inferlib_macros::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Keep printing 'hello, world!' 100 times.".to_string());
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    let start = Instant::now();

    let model = Model::get_auto();
    let eos_tokens = model.eos_tokens();
    let tokenizer = model.get_tokenizer();

    let ctx = Context::new(&model);
    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let cache_table = CacheTable::new(
        LEADER_CAPACITY,
        FOLLOWER_CAPACITY,
        LEADER_LEN as u32,
        FOLLOWER_LEN as u32,
    );
    let mut prev_window = vec![0u32; LEADER_LEN + FOLLOWER_LEN - 1];
    update_cache(&cache_table, &mut prev_window, &ctx.get_token_ids());

    let mut all_generated_tokens = Vec::new();
    let mut num_tokens_per_step = Vec::new();

    println!("Starting generation with speculative decoding...");

    loop {
        let (draft_tokens, draft_pos_ids) = draft(&cache_table, &prev_window);
        let accepted = ctx.verify_draft(&draft_tokens, &draft_pos_ids);

        update_cache(&cache_table, &mut prev_window, &accepted);
        num_tokens_per_step.push(accepted.len());
        all_generated_tokens.extend_from_slice(&accepted);

        if all_generated_tokens.len() >= max_num_outputs
            || eos_tokens
                .iter()
                .any(|seq| all_generated_tokens.ends_with(seq))
        {
            break;
        }
    }

    let output = tokenizer.detokenize(&all_generated_tokens);

    println!("Generation completed.");

    println!(
        "Output: {:?} (total elapsed: {:?})",
        output,
        start.elapsed()
    );

    if !all_generated_tokens.is_empty() {
        println!(
            "Per token latency: {:?}, Mean accepted tokens per step: {:.4}",
            start.elapsed() / (all_generated_tokens.len() as u32),
            num_tokens_per_step.iter().sum::<usize>() as f64 / num_tokens_per_step.len() as f64
        );
    }

    Ok(())
}
