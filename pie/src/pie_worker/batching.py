"""
Batch management for PIE backend inference.

This module provides the Batch dataclass that holds inference batch state
and handles tensor creation and response packaging.
"""

from __future__ import annotations

from typing import Any

import time
import numpy as np
import torch
from numba import njit, prange

from . import message


class Batch:
    """
    Holds the accumulated state for a specific inference step and handles packaging.

    This consolidates the state storage (formerly BatchState) and packaging logic
    (formerly ResponsePackager) into a single unified class.
    """

    def __init__(
        self,
        args: dict[str, Any],
        kv_page_size: int,
        max_dist_size: int,
        adapters: dict[int, Any],
    ) -> None:
        """
        Initialize a Batch from BatchedForwardPassRequest dict.

        Args:
            args: Dictionary with batched request fields
            kv_page_size: KV cache page size from model config
            max_dist_size: Max distribution size from model config
            adapters: Dictionary of active adapters
        """
        # Initialize timing dict as instance attribute
        self.timing: dict[str, float] = {
            "decode_u32": 0.0,
            "mask_loop": 0.0,
            "brle_decode": 0.0,
            "sampler_loop": 0.0,
        }

        # Initialize mutable containers
        self.adapter_indices: list[int] = []
        self.adapter_seeds: list[int] = []
        self.adapter_subpass_needed: bool = False
        self.indices_for_logits: list[int] = []
        self.indices_for_embed_storage: list[int] = []
        self.embed_storage_pointers: list[int] = []

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
        self.token_ids = _decode_i64(args["token_ids"])

        self.position_ids = _decode_u32(args["position_ids"]).astype(np.int32)
        self.kv_page_indices = _decode_u32(args["kv_page_indices"]).astype(np.int32)
        self.kv_page_indptr = _decode_u32(args["kv_page_indptr"]).astype(np.int32)
        self.kv_last_page_lens = _decode_u32(args["kv_last_page_lens"]).astype(np.int32)
        self.qo_indptr = _decode_u32(args["qo_indptr"]).astype(np.int32)
        self.single_token_mode = args["single_token_mode"]
        self.total_tokens = int(len(self.token_ids))
        self.timing["decode_u32"] = time.perf_counter() - t0

        # ===== VECTORIZED BATCH METADATA GENERATION =====
        t0 = time.perf_counter()

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

        # [OPTIMIZATION] Vectorized computation of per-request token counts
        req_token_counts = np.diff(self.qo_indptr)

        # [OPTIMIZATION] Vectorized sequence length calculation
        # seq_len = (num_pages - 1) * kv_page_size + last_len for num_pages > 0
        num_pages = np.diff(self.kv_page_indptr)
        seq_lens = np.where(
            num_pages > 0,
            (num_pages - 1) * kv_page_size + self.kv_last_page_lens,
            self.kv_last_page_lens,
        )

        # [OPTIMIZATION] Vectorized repeat - No Python loops!
        all_seq_lens = np.repeat(seq_lens, req_token_counts)

        # [OPTIMIZATION] Vectorized position IDs calculation
        # Formula: global_index - request_start + context_len
        context_lens = seq_lens - req_token_counts
        global_indices = np.arange(self.total_tokens, dtype=np.int32)
        request_starts = np.repeat(self.qo_indptr[:-1], req_token_counts)
        request_contexts = np.repeat(context_lens, req_token_counts)
        position_ids_np = global_indices - request_starts + request_contexts

        # [OPTIMIZATION] Vectorized cumsum for bit offsets
        token_acc_seq_lens_np = np.zeros(self.total_tokens + 1, dtype=np.int32)
        np.cumsum(all_seq_lens, out=token_acc_seq_lens_np[1:])

        self.timing["mask_loop"] = time.perf_counter() - t0

        # Call batch decoder ONCE for all tokens
        t_brle = time.perf_counter()
        self.attention_masks = decode_brle_batch(
            flattened_masks_u32, mask_indptr, position_ids_np, token_acc_seq_lens_np
        )
        self.timing["brle_decode"] = time.perf_counter() - t_brle

        # ===== PRE-COMPUTE SAMPLER ARRAY SIZES =====
        t0 = time.perf_counter()

        # Count total output samplers for pre-allocation
        total_samplers = sum(len(s) for s in output_token_samplers)

        # Pre-allocate flat sampler arrays
        temperatures_arr = np.ones(total_samplers, dtype=np.float32)
        top_k_arr = np.zeros(total_samplers, dtype=np.int32)
        top_p_arr = np.ones(total_samplers, dtype=np.float32)
        min_p_arr = np.zeros(total_samplers, dtype=np.float32)
        sampler_types_list = []

        # Pre-allocate request output counts
        request_output_counts_arr = np.empty(num_requests, dtype=np.int32)

        sampler_cursor = 0

        # ===== PER-REQUEST LOOP (adapter/sampler/response setup) =====
        token_offset = 0

        for i in range(num_requests):
            req_token_count = int(req_token_counts[i])

            # Handle adapters
            adapter_idx = adapter_indices[i]
            if adapter_idx is not None and adapter_idx in adapters:
                seed = adapter_seeds[i] if adapter_seeds[i] is not None else 0
                self.adapter_seeds.extend([seed] * req_token_count)
                self.adapter_indices.append(adapter_idx)
                self.adapter_subpass_needed = True

            # Handle output indices (adjust for batch offset)
            current_token_offset = int(token_offset)
            for idx in output_token_indices[i]:
                self.indices_for_logits.append(idx + current_token_offset)

            # Handle samplers - fill pre-allocated arrays
            req_samplers = output_token_samplers[i]
            num_outputs = len(req_samplers)
            request_output_counts_arr[i] = num_outputs

            for sampler_config in req_samplers:
                sampler_idx = sampler_config.get("sampler", 1)
                sampler_types_list.append(sampler_idx)

                # Fill flat arrays directly (no dict allocation)
                temperatures_arr[sampler_cursor] = sampler_config.get(
                    "temperature", 1.0
                )

                if sampler_idx == 0:
                    top_k_arr[sampler_cursor] = min(
                        sampler_config.get("top_k", max_dist_size),
                        max_dist_size,
                    )
                else:
                    top_k_arr[sampler_cursor] = sampler_config.get("top_k", 0)
                    top_p_arr[sampler_cursor] = sampler_config.get("top_p", 1.0)
                    min_p_arr[sampler_cursor] = sampler_config.get("min_p", 0.0)

                sampler_cursor += 1

            token_offset += req_token_count

        self.timing["sampler_loop"] = time.perf_counter() - t0

        # Store pre-allocated arrays
        self.sampler_types = sampler_types_list
        self.temperatures = temperatures_arr
        self.top_k_values = top_k_arr
        self.top_p_values = top_p_arr
        self.min_p_values = min_p_arr
        self.request_output_counts = request_output_counts_arr

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

        # Vectorized tensor creation from NumPy arrays (no list comprehension)
        temperatures = (
            torch.as_tensor(self.temperatures, device=device, dtype=dtype)
            .clamp(min=1e-6)
            .unsqueeze(1)
        )

        # Pre-build sampler param tensors (avoid per-group construction)
        top_k_tensor = torch.as_tensor(
            self.top_k_values, device=device, dtype=torch.long
        )
        top_p_tensor = torch.as_tensor(self.top_p_values, device=device, dtype=dtype)
        min_p_tensor = torch.as_tensor(self.min_p_values, device=device, dtype=dtype)

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
            "top_k": top_k_tensor,
            "top_p": top_p_tensor,
            "min_p": min_p_tensor,
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
        num_requests = len(self.request_output_counts)

        # Early return if no logits needed
        if not self.indices_for_logits:
            return [
                message.ForwardPassResponse(dists=[], tokens=[])
                for _ in range(num_requests)
            ]

        final_dists = sampling_results["dists"]
        final_tokens_list = sampling_results["tokens"]

        responses = []
        cursor = 0

        for req_idx in range(num_requests):
            num_outputs = int(self.request_output_counts[req_idx])
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


@njit(cache=True)
def decode_brle_batch(
    flattened_masks: np.ndarray,
    mask_indptr: np.ndarray,
    position_ids: np.ndarray,
    token_acc_seq_lens: np.ndarray,
) -> np.ndarray:
    """
    Decode BRLE masks for an entire batch using Numba JIT.

    Optimized with slice assignment which Numba compiles to SIMD/memset
    block writes - faster than parallel (prange) due to thread pool overhead.

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
                # Slice assignment compiles to SIMD/memset
                result[curr_bit_pos : curr_bit_pos + eff_len] = True

            bits_consumed += eff_len
            curr_bit_pos += eff_len
            is_true_run = not is_true_run

    return result
