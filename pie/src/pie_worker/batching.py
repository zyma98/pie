"""
Batch management for PIE backend inference.

This module provides the Batch dataclass that holds inference batch state
and handles tensor creation and response packaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import time
import numpy as np
import torch
from numba import njit, prange

from . import message


@dataclass
class Batch:
    """
    Holds the accumulated state for a specific inference step and handles packaging.

    This consolidates the state storage (formerly BatchState) and packaging logic
    (formerly ResponsePackager) into a single unified class.
    """

    # Input tokens and positions
    token_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    position_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )

    # KV Cache Layout
    kv_page_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    kv_page_indptr: np.ndarray = field(
        default_factory=lambda: np.array([0], dtype=np.int32)
    )
    kv_last_page_lens: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )

    # Query/Output indirection pointers
    qo_indptr: np.ndarray = field(default_factory=lambda: np.array([0], dtype=np.int32))

    # Attention masks (one per request, flattened later)
    attention_masks: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_)
    )

    # Adapters
    adapter_indices: list[int] = field(default_factory=list)
    adapter_seeds: list[int] = field(default_factory=list)
    adapter_subpass_needed: bool = False

    # Output mapping for logits and embeddings
    indices_for_logits: list[int] = field(default_factory=list)
    indices_for_embed_storage: list[int] = field(default_factory=list)
    embed_storage_pointers: list[int] = field(default_factory=list)

    # Sampler configuration
    sampler_types: list[int] = field(default_factory=list)
    sampler_params: list[dict] = field(default_factory=list)

    # Metadata for tracking and reconstruction
    requests: list[message.ForwardPassRequest] = field(default_factory=list)
    total_tokens: int = 0
    single_token_mode: bool = True

    def get_model_inputs(self, device: torch.device) -> dict[str, Any]:
        """
        Finalize batch preparation and create input tensors for the model.

        Args:
            device: The torch device to create tensors on.

        Returns:
            Dictionary containing input tensors for the model engine.
        """

        self.adapter_subpass_needed = False

        return {
            "token_ids": torch.as_tensor(
                self.token_ids, device=device, dtype=torch.long
            ),
            "position_ids": torch.as_tensor(
                self.position_ids, device=device, dtype=torch.int32
            ),
            "qo_indptr": torch.as_tensor(
                self.qo_indptr, device=device, dtype=torch.int32
            ),
            "kv_page_indices": torch.as_tensor(
                self.kv_page_indices, device=device, dtype=torch.int32
            ),
            "kv_page_indptr": torch.as_tensor(
                self.kv_page_indptr, device=device, dtype=torch.int32
            ),
            "kv_last_page_lens": torch.as_tensor(
                self.kv_last_page_lens, device=device, dtype=torch.int32
            ),
            "custom_mask": torch.as_tensor(
                self.attention_masks, device=device, dtype=torch.bool
            ),
            "single_token_inference_mode": self.single_token_mode,
            "adapter_indices": (
                self.adapter_indices if self.adapter_subpass_needed else []
            ),
            "adapter_seeds": (
                torch.as_tensor(self.adapter_seeds, device=device, dtype=torch.long)
                if self.adapter_subpass_needed
                else None
            ),
            "total_pages_cpu": self.kv_page_indptr[-1],
        }

    def get_sampling_metadata(
        self, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Any]:
        """
        Prepare the metadata required for the SamplingPass.

        Args:
            device: Torch device.
            dtype: Torch dtype for temperatures.

        Returns:
            Dictionary containing sampling metadata.
        """
        # Return empty if no logits needed
        if not self.indices_for_logits:
            return {"indices_for_logits": None}

        indices_for_logits = self.indices_for_logits

        temperatures = (
            torch.tensor(
                [p["temperature"] for p in self.sampler_params],
                device=device,
                dtype=dtype,
            )
            .clamp(min=1e-6)
            .unsqueeze(1)
        )

        # Group samplers
        sampler_groups: dict[int, list[int]] = {}
        for i, sampler_idx in enumerate(self.sampler_types):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        return {
            "indices_for_logits": indices_for_logits,
            "temperatures": temperatures,
            "sampler_groups": sampler_groups,
            "sampler_params": self.sampler_params,
        }

    def create_responses(
        self, sampling_results: dict[str, Any]
    ) -> list[message.ForwardPassResponse]:
        """
        Package the sampling results into responses for each original request.

        Args:
            sampling_results: Dictionary containing 'tokens' and 'dists'.

        Returns:
            List of responses in the order of requests.
        """
        # Early return if no logits needed
        if not self.indices_for_logits:
            return [
                message.ForwardPassResponse(dists=[], tokens=[]) for _ in self.requests
            ]

        final_dists = sampling_results["dists"]
        final_tokens_list = sampling_results["tokens"]

        responses = []
        cursor = 0

        for req in self.requests:
            output_token_indices = req.output_token_indices or []
            num_outputs = len(output_token_indices)
            request_dists = []
            request_tokens = []

            for i in range(cursor, cursor + num_outputs):
                if self.sampler_types[i] == 0:
                    # Distribution request
                    if final_dists[i] is not None:
                        request_dists.append(final_dists[i])
                else:
                    # Sampling request
                    request_tokens.append(final_tokens_list[i])

            responses.append(
                message.ForwardPassResponse(dists=request_dists, tokens=request_tokens)
            )
            cursor += num_outputs

        return responses

    @classmethod
    def from_batched_request(
        cls,
        args: dict[str, Any],
        kv_page_size: int,
        max_dist_size: int,
        adapters: dict[int, Any],
    ) -> tuple[Batch, dict[str, float]]:
        """
        Convert BatchedForwardPassRequest dict to internal Batch object.

        Args:
            args: Dictionary with batched request fields
            kv_page_size: KV cache page size from model config
            max_dist_size: Max distribution size from model config
            adapters: Dictionary of active adapters

        Returns:
            Tuple of (Batch object ready for inference, timing dict)
        """
        timing = {
            "decode_u32": 0.0,
            "mask_loop": 0.0,
            "brle_decode": 0.0,
            "adapter_loop": 0.0,
            "sampler_loop": 0.0,
            "embed_loop": 0.0,
        }

        batch = cls()

        # Helper to decode bytes as u32 array or pass through lists
        def _decode_u32(data):
            if isinstance(data, bytes):
                return np.frombuffer(data, dtype=np.uint32)
            return np.array(data, dtype=np.uint32)

        # Helper to decode bytes as i64 array
        def _decode_i64(data):
            if isinstance(data, bytes):
                # Assume input is u32/i32 for now as input is from msgpack
                # but we need i64 for token_ids in torch
                return np.frombuffer(data, dtype=np.uint32).astype(np.int64)
            return np.array(data, dtype=np.int64)

        # Direct assignments - decode bytes as u32 arrays
        t0 = time.perf_counter()
        # token_ids are long (i64)
        batch.token_ids = _decode_i64(args["token_ids"])

        batch.position_ids = _decode_u32(args["position_ids"]).astype(np.int32)
        batch.kv_page_indices = _decode_u32(args["kv_page_indices"]).astype(np.int32)
        batch.kv_page_indptr = _decode_u32(args["kv_page_indptr"]).astype(np.int32)
        batch.kv_last_page_lens = _decode_u32(args["kv_last_page_lens"]).astype(
            np.int32
        )
        batch.qo_indptr = _decode_u32(args["qo_indptr"]).astype(np.int32)
        batch.single_token_mode = args["single_token_mode"]
        batch.total_tokens = int(len(batch.token_ids))
        timing["decode_u32"] = time.perf_counter() - t0

        # Process per-request data
        flattened_masks_u32 = _decode_u32(args["flattened_masks"]).astype(np.int32)
        mask_indptr = _decode_u32(args["mask_indptr"]).astype(np.int32)

        adapter_indices = args["adapter_indices"]
        adapter_seeds = args["adapter_seeds"]
        output_token_indices = args["output_token_indices"]
        output_token_samplers = args["output_token_samplers"]
        output_embed_ptrs = args["output_embed_ptrs"]
        output_embed_indices = args["output_embed_indices"]

        num_requests = len(adapter_indices)

        # ===== BATCH BRLE DECODING (once for all tokens) =====
        t0 = time.perf_counter()

        # First pass: compute per-token metadata (seq_len, context_len, position_id)
        all_seq_lens = []  # seq_len for each token
        all_position_ids = []  # position_id for each token
        request_token_ranges = []  # (start_token, end_token, seq_len) per request

        token_cursor = 0
        for i in range(num_requests):
            req_token_count = batch.qo_indptr[i + 1] - batch.qo_indptr[i]

            # Calculate sequence length from KV pages
            kv_start = batch.kv_page_indptr[i]
            kv_end = batch.kv_page_indptr[i + 1]
            num_pages = kv_end - kv_start
            kv_last_len = batch.kv_last_page_lens[i]

            if num_pages >= 1:
                seq_len = kv_page_size * (num_pages - 1) + kv_last_len
            else:
                seq_len = kv_last_len

            context_len = seq_len - req_token_count

            request_token_ranges.append(
                (token_cursor, token_cursor + req_token_count, seq_len)
            )

            for j in range(req_token_count):
                all_seq_lens.append(seq_len)
                all_position_ids.append(context_len + j)

            token_cursor += req_token_count

        # Compute token_acc_seq_lens (cumulative bit offsets)
        token_acc_seq_lens = [0]
        for sl in all_seq_lens:
            token_acc_seq_lens.append(token_acc_seq_lens[-1] + sl)

        # Prepare arrays for batch decoder
        # Already numpy arrays from _decode_u32 (casted to int32 above)
        flattened_np = flattened_masks_u32
        mask_indptr_np = mask_indptr
        position_ids_np = np.array(all_position_ids, dtype=np.int32)
        token_acc_seq_lens_np = np.array(token_acc_seq_lens, dtype=np.int32)

        # Call batch decoder ONCE for all tokens
        t_brle = time.perf_counter()
        batch.attention_masks = decode_brle_batch(
            flattened_np, mask_indptr_np, position_ids_np, token_acc_seq_lens_np
        )
        timing["brle_decode"] = time.perf_counter() - t_brle

        # Unpack all bits
        timing["mask_loop"] = time.perf_counter() - t0

        # ===== PER-REQUEST LOOP (slicing pre-decoded masks) =====
        token_offset = 0

        for i in range(num_requests):
            start_token, end_token, seq_len = request_token_ranges[i]
            req_token_count = end_token - start_token

            # Handle adapters
            t0 = time.perf_counter()
            adapter_idx = adapter_indices[i]
            if adapter_idx is not None and adapter_idx in adapters:
                seed = adapter_seeds[i] if adapter_seeds[i] is not None else 0
                batch.adapter_seeds.extend([seed] * req_token_count)
                batch.adapter_indices.append(adapter_idx)
                batch.adapter_subpass_needed = True

            # Handle output indices (adjust for batch offset)
            # Ensure token_offset is standard python int to avoid numpy scalar issues in indices_for_logits
            current_token_offset = int(token_offset)
            for idx in output_token_indices[i]:
                batch.indices_for_logits.append(idx + current_token_offset)
            timing["adapter_loop"] += time.perf_counter() - t0

            # Handle samplers
            t0 = time.perf_counter()
            for sampler_config in output_token_samplers[i]:
                params = {}
                sampler_idx = sampler_config.get("sampler", 1)
                batch.sampler_types.append(sampler_idx)

                if sampler_idx == 0:
                    params["top_k"] = min(
                        sampler_config.get("top_k", max_dist_size),
                        max_dist_size,
                    )
                else:
                    params["top_k"] = sampler_config.get("top_k", 0)
                    params["top_p"] = sampler_config.get("top_p", 1.0)
                    params["min_p"] = sampler_config.get("min_p", 0.0)

                params["temperature"] = sampler_config.get("temperature", 1.0)
                batch.sampler_params.append(params)
            timing["sampler_loop"] += time.perf_counter() - t0

            # Handle embed outputs
            t0 = time.perf_counter()
            # for idx, ptr in zip(output_embed_indices[i], output_embed_ptrs[i]):
            #     batch.indices_for_embed_storage.append(idx + token_offset)
            #     batch.embed_storage_pointers.append(ptr)

            # Create dummy request for response packaging
            dummy_req = message.ForwardPassRequest(
                input_tokens=[],
                input_token_positions=[],
                input_embed_ptrs=[],
                input_embed_positions=[],
                adapter=adapter_indices[i],
                adapter_seed=adapter_seeds[i],
                mask=[],
                output_token_indices=output_token_indices[i],
                output_token_samplers=output_token_samplers[i],
            )
            batch.requests.append(dummy_req)
            timing["embed_loop"] += time.perf_counter() - t0

            token_offset += req_token_count

        return batch, timing


@njit(parallel=False, cache=False)
def decode_brle_batch(
    flattened_masks: np.ndarray,
    mask_indptr: np.ndarray,
    position_ids: np.ndarray,
    token_acc_seq_lens: np.ndarray,
) -> np.ndarray:
    """
    Decode BRLE masks for an entire batch using Numba JIT.

    Args:
        flattened_masks: Concatenated BRLE run lengths (int32)
        mask_indptr: Pointers to BRLE ranges per token (int32)
        position_ids: Position of each token, defines valid_len (int32)
        token_acc_seq_lens: Cumulative bit offsets per token (int32)

    Returns:
        Flat boolean array with all mask values
    """
    num_tokens = len(position_ids)
    total_bits = token_acc_seq_lens[-1]
    result = np.zeros(total_bits, dtype=np.bool_)

    for k in range(num_tokens):
        rle_start = mask_indptr[k]
        rle_end = mask_indptr[k + 1]
        global_bit_start = token_acc_seq_lens[k]
        valid_len = position_ids[k] + 1

        curr_bit_pos = global_bit_start
        bits_consumed = 0
        is_true_run = True

        for run_idx in range(rle_start, rle_end):
            if bits_consumed >= valid_len:
                break

            run_len = flattened_masks[run_idx]
            remaining = valid_len - bits_consumed
            eff_len = min(run_len, remaining)

            if is_true_run and eff_len > 0:
                for bit_off in range(eff_len):
                    result[curr_bit_pos + bit_off] = True

            bits_consumed += eff_len
            curr_bit_pos += eff_len
            is_true_run = not is_true_run

    return result
