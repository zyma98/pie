import asyncio

# import csv  # W&B CHANGE: no longer needed (seed-score artifacts removed)
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Dict
from contextlib import AsyncExitStack
import html  # W&B CHANGE: for wandb.Html logging
from collections import defaultdict, deque

from blake3 import blake3
import numpy as np
import wandb  # Weights & Biases

# progress bars
from tqdm.auto import tqdm

# Assume pie is an installed library with this structure
# You may need to adjust imports based on the actual library
from pie import PieClient, Instance, Event, compile_program
# Use the refactored imports
from countdown import CountdownTasksDataset, reward_function, SYSTEM_MESSAGE, USER_TEMPLATE, RESPONSE_PROMPT

# ==============================================================================
# Configuration
# ==============================================================================

# --- Logging Verbosity ---
# Toggle to re-enable chatty per-worker prints if desired.
VERBOSE_WORKER_LOGS = True

# --- Server and Paths ---
SERVER_URIS = [
    "ws://127.0.0.1:8080",
    # "ws://127.0.0.1:8081",
]
SCRIPT_DIR = Path(__file__).resolve().parent
# NOTE: Update this path to your actual inferlet dependency location
INFERLET_DEPS = [
    'inferlet={ path = "/home/ingim/Workspace/pie/inferlet" }',
    'inferlet-macros={ path = "/home/ingim/Workspace/pie/inferlet-macros" }',
    'pico-args = "0.5.0"',
    'futures = "0.3.31"',
    'serde = { version = "1.0", features = ["derive"] }',
    'serde_json = "1.0"',
]
INFERLET_SRC_FILES = [
    "es-init.rs",
    "es-rollout.rs",
    "es-update.rs",
]

# --- ES Hyperparameters ---
ADAPTER_NAME = "evo-countdown-v1"
TRAINING_STEPS = 10000
POPULATION_SIZE = 512 * 2  # Total seeds per step across all clients
TASKS_PER_SEED = 4  # Number of tasks to evaluate for each seed

# Each inferlet handles this many (seed, task) requests concurrently.
NUM_ROLLOUTS_PER_WORKER = 8

LORA_RANK = 8
LORA_ALPHA = 16.0
INITIAL_SIGMA = 0.005
MAX_SIGMA = 0.014
MU_FRACTION = 0.5
MAX_TOKENS_GEN = 512  # tokens per request

# >>> NEW: Evaluation Configuration
EVAL_EVERY_N_STEPS = 20  # How often to run evaluation on the test set
EVAL_TASKS_PER_WORKER = 16 # Batch size for evaluation tasks per inferlet

# --- Per-server token budgets (tokens available for generation across concurrent workers).
DEFAULT_MAX_TOKENS_PER_SERVER = 512 * (198)
# You can override specific servers here; any URI not present uses DEFAULT_MAX_TOKENS_PER_SERVER.
MAX_TOKENS_PER_SERVER = {
    uri: DEFAULT_MAX_TOKENS_PER_SERVER for uri in SERVER_URIS
}

# --- Dataset Config ---
DATA_PATH = "./Countdown-Tasks-3to4"

# --- W&B Config ---
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "pie-es-v4")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_MODE = os.getenv("WANDB_MODE")  # e.g., "offline"
WANDB_TAGS = ["es", "countdown", "lora"]


# ==============================================================================
# Inferlet Management & Execution
# ==============================================================================

def cache_paths_for_file(src_path: Path, base_dir: Path):
    """Computes cache filenames using the BLAKE3 hash of the file contents."""
    code_bytes = src_path.read_bytes()
    code_hash = blake3(code_bytes).hexdigest()
    wasm_path = base_dir / f"program-{code_hash}.wasm"
    hash_path = base_dir / f"program-{code_hash}.hash"
    return code_hash, wasm_path, hash_path


async def get_wasm_from_cache_or_compile(src_path: Path, deps: list[str], base_dir: Path) -> bytes:
    """Reuses cached WASM or compiles the Rust source file using the provided client."""
    if not src_path.exists():
        raise FileNotFoundError(f"Rust source file not found: {src_path}")

    code_hash, wasm_path, hash_path = cache_paths_for_file(src_path, base_dir)

    if wasm_path.exists():
        tqdm.write(f"[Cache] Reusing cached wasm for {src_path.name} (hash {code_hash[:8]})")
        return wasm_path.read_bytes()

    tqdm.write(f"[Compile] No cache for {src_path.name}...")
    program_bytes = await compile_program(str(src_path), deps)
    wasm_path.write_bytes(program_bytes)
    hash_path.write_text(code_hash + "\n")
    tqdm.write(f"[Compile] Saved wasm to {wasm_path.name}")
    return program_bytes


async def launch_and_get_result(
        client: PieClient,
        program_hash: str,
        arguments: list[str],
        worker_id: Any = 0,
        verbose: bool = False,
) -> Optional[str]:
    """Launches an inferlet and waits for it to terminate, returning the final message."""
    if verbose:
        tqdm.write(f"🚀 Worker {worker_id}: Launching instance with hash {program_hash[:8]}...")
    instance = await client.launch_instance(program_hash, arguments=arguments)
    final_payload = None
    while True:
        event, message = await instance.recv()

        match event:
            case Event.Completed:
                if verbose:
                    tqdm.write(f"✅ Worker {worker_id}: Instance {instance.instance_id} finished. Reason: {message}")
                break
            case Event.Message:
                final_payload = message
            case Event.Aborted:
                if verbose:
                    tqdm.write(f"  [Worker {worker_id} / {instance.instance_id} / {event}] {message}")
            case Event.Exception:
                if verbose:
                    tqdm.write(f"  [Worker {worker_id} / {instance.instance_id} / {event}] {message}")
            case Event.OutOfResources:
                if verbose:
                    tqdm.write(f"  [Worker {worker_id} / {instance.instance_id} / {event}] {message}")

    return final_payload


# ==============================================================================
# Adaptive Rollout Scheduling (Cross-Client)
# ==============================================================================

def _compute_client_capacity(server_uri: str) -> int:
    """Max concurrent workers for a client based on its token budget."""
    budget = MAX_TOKENS_PER_SERVER.get(server_uri, DEFAULT_MAX_TOKENS_PER_SERVER)
    denom = MAX_TOKENS_GEN * NUM_ROLLOUTS_PER_WORKER
    cap = max(1, budget // max(1, denom))
    return int(cap)


async def _rollout_batch_with_retry(
        client: PieClient,
        program_hash: str,
        adapter_name: str,
        seeds_slice: np.ndarray,
        tasks_slice: np.ndarray,
        who: str,
        max_retries: int = 1,
):
    """Run one inferlet on a batch; retry once on failure/mismatch. Returns list[str] or None."""
    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        worker_tasks_json = json.dumps([item["prompt"] for item in tasks_slice])
        rollout_args = [
            "--name", adapter_name,
            "--seeds", ",".join(map(str, seeds_slice)),
            "--tasks-json", worker_tasks_json,
            "--max-num-outputs", str(MAX_TOKENS_GEN),
        ]
        res_json = await launch_and_get_result(
            client, program_hash, rollout_args, worker_id=f"{who}-try{attempt}", verbose=VERBOSE_WORKER_LOGS
        )
        if res_json:
            try:
                texts = json.loads(res_json)
            except Exception:
                texts = None
            if isinstance(texts, list) and len(texts) == len(seeds_slice):
                return texts  # success
    return None


async def _pop_batch_from_queue(
        q: deque,
        q_lock: asyncio.Lock,
        batch_size: int,
):
    """Atomically pop up to batch_size items from the global queue."""
    async with q_lock:
        if not q:
            return None, None
        seeds_list = []
        tasks_list = []
        for _ in range(min(batch_size, len(q))):
            s, t = q.popleft()
            seeds_list.append(int(s))
            tasks_list.append(t)
    return np.array(seeds_list, dtype=np.int64), np.array(tasks_list, dtype=object)


async def _process_client_rollouts_cross_client(
        client: PieClient,
        server_uri: str,
        program_hash: str,
        adapter_name: str,
        global_queue: deque,
        queue_lock: asyncio.Lock,
        client_idx: Any,
        pbar: Optional[tqdm] = None,
        batch_size: int = 8,  # >>> CHANGED: Add batch_size argument
):
    """Adaptive cross-client scheduler using a shared global queue."""
    capacity = _compute_client_capacity(server_uri)
    tqdm.write(f"Client {client_idx} @ {server_uri}: capacity={capacity}, batch_size={batch_size}")

    generated_texts: list[str] = []
    generated_seeds: list[int] = []
    generated_tasks: list[dict] = []

    running_tasks = set()
    task_meta = {}
    issued = 0

    # Prime initial slots
    while len(running_tasks) < capacity:
        seeds_slice, tasks_slice = await _pop_batch_from_queue(global_queue, queue_lock, batch_size)
        if seeds_slice is None:
            break
        who = f"C{client_idx}-B{issued}"
        issued += 1
        t = asyncio.create_task(_rollout_batch_with_retry(
            client, program_hash, adapter_name, seeds_slice, tasks_slice, who
        ))
        running_tasks.add(t)
        task_meta[t] = (seeds_slice, tasks_slice, who)

    # Keep pipeline full while there is work or in-flight tasks
    while running_tasks:
        done, pending = await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            seeds_slice, tasks_slice, who = task_meta.pop(t)
            texts = t.result()
            if texts is not None:
                generated_texts.extend(texts)
                generated_seeds.extend(list(map(int, seeds_slice)))
                generated_tasks.extend(list(tasks_slice))

            if pbar is not None:
                try:
                    pbar.update(len(seeds_slice))
                except Exception:
                    pass

            # Refill this slot from the global queue
            seeds_slice2, tasks_slice2 = await _pop_batch_from_queue(global_queue, queue_lock, batch_size)
            if seeds_slice2 is not None:
                who2 = f"C{client_idx}-B{issued}"
                issued += 1
                t2 = asyncio.create_task(_rollout_batch_with_retry(
                    client, program_hash, adapter_name, seeds_slice2, tasks_slice2, who2
                ))
                pending.add(t2)
                task_meta[t2] = (seeds_slice2, tasks_slice2, who2)
        running_tasks = set(pending)

    return generated_texts, generated_seeds, generated_tasks, capacity, server_uri


# ==============================================================================
# >>> NEW: Evaluation Logic
# ==============================================================================

async def run_evaluation(
        clients: list[PieClient],
        server_uris: list[str],
        program_hash: str,
        adapter_name: str,
        eval_dataset: CountdownTasksDataset,
        step: int,
) -> Dict[str, Any]:
    """Runs evaluation on the central model parameters (seed=0) across all clients."""
    tqdm.write("\n" + "-" * 20 + f" Running Evaluation @ Step {step} " + "-" * 20)
    eval_start_time = time.time()

    # --- 1. Prepare evaluation tasks (all with seed=0) ---
    num_eval_tasks = len(eval_dataset)
    eval_seeds = np.zeros(num_eval_tasks, dtype=np.int64)  # All seeds are 0 for the central model
    all_eval_tasks = [eval_dataset[i] for i in range(num_eval_tasks)]

    global_queue = deque((int(s), t) for s, t in zip(eval_seeds.tolist(), all_eval_tasks))
    queue_lock = asyncio.Lock()

    # --- 2. Run rollout phase across all clients ---
    generated_texts: list[str] = []
    generated_tasks: list[dict] = []

    client_manager_tasks = []
    with tqdm(
            total=num_eval_tasks,
            desc=f"Evaluation @ Step {step}",
            dynamic_ncols=True,
            leave=False,
    ) as pbar:
        for client_idx, (client, server_uri) in enumerate(zip(clients, server_uris)):
            t = asyncio.create_task(_process_client_rollouts_cross_client(
                client=client,
                server_uri=server_uri,
                program_hash=program_hash,
                adapter_name=adapter_name,
                global_queue=global_queue,
                queue_lock=queue_lock,
                client_idx=f"Eval-C{client_idx}",
                pbar=pbar,
                batch_size=EVAL_TASKS_PER_WORKER,
            ))
            client_manager_tasks.append(t)

        async for completed in _as_completed_iter(client_manager_tasks):
            texts_i, _, tasks_i, _, _ = await completed
            generated_texts.extend(texts_i)
            generated_tasks.extend(tasks_i)

    # --- 3. Score the generations ---
    reward_infos = [
        reward_function(text, generated_tasks[i]["nums"], generated_tasks[i]["target"], end_token="<|eot_id|>")
        for i, text in enumerate(generated_texts)
    ]
    scores = [float(ri.get("reward", 0.0)) for ri in reward_infos]
    answer_rewards = [float(ri.get("answer_reward", 0.0)) for ri in reward_infos]

    # --- 4. Aggregate and return metrics ---
    eval_mean_reward = float(np.mean(scores)) if scores else 0.0
    eval_success_rate = float(np.mean(answer_rewards)) if answer_rewards else 0.0
    eval_duration = time.time() - eval_start_time

    tqdm.write(
        f"✅ Evaluation Complete: mean_reward={eval_mean_reward:.4f}, "
        f"success_rate={eval_success_rate:.3f}, duration={eval_duration:.1f}s"
    )

    # Log a few examples to W&B
    num_to_log = min(10, len(generated_texts))
    eval_html = "<pre>No evaluation predictions to log.</pre>"
    if num_to_log > 0:
        indices = np.random.choice(len(generated_texts), size=num_to_log, replace=False)
        table_rows = [
            '<tr><th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Prompt</th>'
            '<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Generation</th>'
            '<th style="border: 1px solid #ddd; padding: 8px;">Reward</th>'
            '<th style="border: 1px solid #ddd; padding: 8px;">Success</th></tr>'
        ]
        for idx in indices:
            prompt_html = f"<pre>nums = {generated_tasks[idx]['nums']}\ntarget = {generated_tasks[idx]['target']}</pre>"
            text_html = f"<pre>{html.escape(generated_texts[idx])}</pre>"
            reward = scores[idx]
            success_icon = "✅" if answer_rewards[idx] == 1.0 else "❌"
            table_rows.append(
                f'<tr><td style="border: 1px solid #ddd; padding: 8px;">{prompt_html}</td>'
                f'<td style="border: 1px solid #ddd; padding: 8px;">{text_html}</td>'
                f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{reward:.3f}</td>'
                f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{success_icon}</td></tr>'
            )
        eval_html = f'<table style="width:100%; border-collapse: collapse;">{"".join(table_rows)}</table>'

    return {
        "eval/mean_reward": eval_mean_reward,
        "eval/success_rate": eval_success_rate,
        "eval/duration_seconds": eval_duration,
        "eval/examples": wandb.Html(eval_html),
    }


# ==============================================================================
# Main Training Script
# ==============================================================================

async def main():
    """The main entry point for the ES training process."""

    # ---------------------------
    # W&B: initialize the run
    # ---------------------------
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{ADAPTER_NAME}-{int(time.time())}",
        job_type="training",
        tags=WANDB_TAGS,
        config={
            "adapter_name": ADAPTER_NAME,
            "training_steps": TRAINING_STEPS,
            "population_size": POPULATION_SIZE,
            "tasks_per_seed": TASKS_PER_SEED,
            "num_rollouts_per_worker": NUM_ROLLOUTS_PER_WORKER,
            "default_max_tokens_per_server": DEFAULT_MAX_TOKENS_PER_SERVER,
            "max_tokens_per_server": MAX_TOKENS_PER_SERVER,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "initial_sigma": INITIAL_SIGMA,
            "mu_fraction": MU_FRACTION,
            "max_sigma": MAX_SIGMA,
            "max_tokens_gen": MAX_TOKENS_GEN,
            "server_uris": SERVER_URIS,
            "data_path": DATA_PATH,
            "eval_every_n_steps": EVAL_EVERY_N_STEPS, # >>> NEW
            "eval_tasks_per_worker": EVAL_TASKS_PER_WORKER, # >>> NEW
        },
        save_code=True,
        reinit=False,
        settings=wandb.Settings(start_method="thread", _disable_stats=False),
        mode=WANDB_MODE if WANDB_MODE else None,
    )
    # >>> CHANGED: Add evaluation metrics
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")

    try:
        try:
            wandb.run.log_code(root=str(SCRIPT_DIR))
        except Exception:
            pass

        async with AsyncExitStack() as stack:
            clients = [await stack.enter_async_context(PieClient(uri)) for uri in SERVER_URIS]
            num_clients = len(clients)
            tqdm.write(f"✅ Connected to {num_clients} Pie server(s).")
            wandb.config.update({"num_clients": num_clients}, allow_val_change=True)

            tqdm.write("\nCompiling and uploading inferlets to all clients...")
            program_hashes = {}
            for src_filename in INFERLET_SRC_FILES:
                src_path = SCRIPT_DIR / src_filename
                program_bytes = await get_wasm_from_cache_or_compile(src_path, INFERLET_DEPS, SCRIPT_DIR)
                program_hash = blake3(program_bytes).hexdigest()
                upload_tasks = [
                    client.upload_program(program_bytes)
                    for client in clients if not await client.program_exists(program_hash)
                ]
                if upload_tasks:
                    tqdm.write(f"Uploading program {src_filename} ({program_hash[:8]})...")
                    await asyncio.gather(*upload_tasks)
                    tqdm.write(f"Program {src_filename} uploaded to all necessary clients.")
                program_hashes[src_path.stem] = program_hash

            tqdm.write("\n" + "=" * 50)
            tqdm.write("Step 0: Initializing ES Adapter on all clients")
            tqdm.write("=" * 50)
            init_args = [
                "--name", ADAPTER_NAME, "--rank", str(LORA_RANK), "--alpha", str(LORA_ALPHA),
                "--population-size", str(POPULATION_SIZE), "--mu-fraction", str(MU_FRACTION),
                "--initial-sigma", str(INITIAL_SIGMA),
            ]
            init_tasks = [
                launch_and_get_result(client, program_hashes["es-init"], init_args, worker_id=f"C{i}-Init")
                for i, client in enumerate(clients)
            ]
            await asyncio.gather(*init_tasks)
            tqdm.write("✅ Adapter initialized on all clients.")

            # --- 3. Load Datasets ---
            dataset = CountdownTasksDataset(DATA_PATH, "train", 100)
            # >>> NEW: Load evaluation dataset
            eval_dataset = CountdownTasksDataset(DATA_PATH, "test")
            wandb.config.update({
                "dataset_size_train_view": len(dataset),
                "dataset_size_eval": len(eval_dataset),
            }, allow_val_change=True)

            # --- 4. Start Training Loop ---
            tqdm.write("\n" + "=" * 50)
            tqdm.write(f"Starting ES Training Loop with {num_clients} clients; adaptive cross-client concurrency.")
            tqdm.write("=" * 50)

            for step in range(1, TRAINING_STEPS + 1):
                start_time = time.time()
                tqdm.write(f"\n--- Step {step}/{TRAINING_STEPS} ---")

                # --- 4a. Rollout Phase (Distributed) ---
                tqdm.write(f"Phase: Rollout across {num_clients} clients (adaptive, shared queue)")
                base_seeds = np.random.randint(-2 ** 63, 2 ** 63 - 1, size=POPULATION_SIZE, dtype=np.int64)
                num_total_tasks = POPULATION_SIZE * TASKS_PER_SEED
                task_idx_matrix = np.random.choice(len(dataset), size=(POPULATION_SIZE, TASKS_PER_SEED), replace=True)
                all_tasks = [dataset[i] for i in task_idx_matrix.reshape(-1)]
                repeated_seeds = np.repeat(base_seeds, TASKS_PER_SEED)

                global_queue = deque((int(s), t) for s, t in zip(repeated_seeds.tolist(), all_tasks))
                queue_lock = asyncio.Lock()

                generated_texts, generated_seeds, generated_tasks, cap_map = [], [], [], {}
                client_manager_tasks = []
                with tqdm(total=num_total_tasks, desc=f"Step {step} rollout", dynamic_ncols=True, leave=False) as pbar:
                    for client_idx, (client, server_uri) in enumerate(zip(clients, SERVER_URIS)):
                        t = asyncio.create_task(_process_client_rollouts_cross_client(
                            client=client, server_uri=server_uri, program_hash=program_hashes["es-rollout"],
                            adapter_name=ADAPTER_NAME, global_queue=global_queue, queue_lock=queue_lock,
                            client_idx=client_idx, pbar=pbar, batch_size=NUM_ROLLOUTS_PER_WORKER,
                        ))
                        client_manager_tasks.append(t)

                    async for completed in _as_completed_iter(client_manager_tasks):
                        texts_i, seeds_i, tasks_i, cap_i, uri_i = await completed
                        generated_texts.extend(texts_i)
                        generated_seeds.extend(seeds_i)
                        generated_tasks.extend(tasks_i)
                        cap_map[uri_i] = cap_i

                out_lens_ws_tokens = [len(t.split()) for t in generated_texts]
                avg_len_ws_tokens = float(np.mean(out_lens_ws_tokens)) if out_lens_ws_tokens else 0.0

                # --- 4b. Scoring & 4c. Aggregation ---
                tqdm.write("Phase: Scoring & Aggregating")
                reward_infos = [
                    reward_function(text, generated_tasks[i]["nums"], generated_tasks[i]["target"], end_token="<|eot_id|>")
                    for i, text in enumerate(generated_texts)
                ]
                scores = [float(ri.get("reward", 0.0)) for ri in reward_infos]
                answer_rewards = [float(ri.get("answer_reward", 0.0)) for ri in reward_infos]

                scores_by_seed = defaultdict(list)
                for s, sc in zip(generated_seeds, scores):
                    scores_by_seed[int(s)].append(float(sc))

                missing_seeds, aggregated_scores = 0, []
                for s in base_seeds:
                    vals = scores_by_seed.get(int(s))
                    aggregated_scores.append(float(np.mean(vals)) if vals else 0.0)
                    if not vals: missing_seeds += 1

                step_mean, step_std = float(np.mean(scores)), float(np.std(scores))
                step_max, step_min = float(np.max(scores)), float(np.min(scores))
                success_rate_train = float(np.mean(answer_rewards)) if answer_rewards else 0.0
                mean_population_score = float(np.mean(aggregated_scores))
                mu_k = max(1, int(np.ceil(MU_FRACTION * POPULATION_SIZE)))
                top_mu_mean = float(np.mean(sorted(aggregated_scores, reverse=True)[:mu_k]))

                # --- 4d. Update Phase (Distributed) ---
                tqdm.write("Phase: Update")
                update_args = [
                    "--name", ADAPTER_NAME, "--seeds", ",".join(map(str, base_seeds)),
                    "--scores", ",".join(map(str, aggregated_scores)), "--max-sigma", str(MAX_SIGMA),
                ]
                update_tasks = [
                    launch_and_get_result(client, program_hashes["es-update"], update_args, worker_id=f"C{i}-Update")
                    for i, client in enumerate(clients)
                ]
                await asyncio.gather(*update_tasks)
                step_duration = time.time() - start_time

                tqdm.write(
                    f"Step {step}: mean={step_mean:.4f} std={step_std:.4f} max={step_max:.4f} "
                    f"succ={success_rate_train:.3f} episodes={len(generated_texts)} dur={step_duration:.1f}s"
                )

                # --- 5. W&B Logging (including periodic evaluation) ---
                metrics_to_log = {
                    "step": step, "mean_reward": step_mean, "std_reward": step_std,
                    "success_rate/train": success_rate_train, "duration_seconds": float(step_duration),
                    "perf/step_duration_sec": float(step_duration), "num_finished_episodes": len(generated_texts),
                    "mean_response_len": float(avg_len_ws_tokens), "es/mean_population_score": mean_population_score,
                    "es/mean_fittest_score": top_mu_mean, "rollout/missing_seeds": int(missing_seeds),
                    "rollout/max_concurrent_workers_per_client": cap_map,
                }

                # >>> NEW: Run evaluation periodically after the update
                if step % EVAL_EVERY_N_STEPS == 0 or step == TRAINING_STEPS:
                    eval_metrics = await run_evaluation(
                        clients=clients, server_uris=SERVER_URIS, program_hash=program_hashes["es-rollout"],
                        adapter_name=ADAPTER_NAME, eval_dataset=eval_dataset, step=step,
                    )
                    metrics_to_log.update(eval_metrics)
                    if step == TRAINING_STEPS: # Save final eval stats to summary
                        wandb.summary["final_eval_success_rate"] = eval_metrics["eval/success_rate"]
                        wandb.summary["final_eval_mean_reward"] = eval_metrics["eval/mean_reward"]

                wandb.log(metrics_to_log, step=step)

            tqdm.write("\nTraining finished!")
            wandb.summary["final_mean_score"] = step_mean
            wandb.summary["final_success_rate_train"] = success_rate_train

    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted by user.")
    finally:
        if wandb.run:
            wandb.finish()


async def _as_completed_iter(tasks):
    """Helper to iterate over tasks as they complete (avoids gather in rollout)."""
    pending = set(tasks)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            yield d


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted by user.")