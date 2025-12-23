from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any

import torch
import torch.nn.functional as fun
import torch.distributed as dist


from . import Spec as SpecBase, Param as ParamBase, Buffer as BufferBase, Config

from ..adapter import AdapterSubpass
from ..quantization import quantize
from ..utils import is_apple_silicon, get_available_memory

if is_apple_silicon():
    import flashinfer_metal as ops  # type: ignore[import-not-found]
else:
    import flashinfer as ops  # type: ignore[import-not-found,no-redef]


def _shard(x, config: Config, row_parallel: bool = False) -> torch.Tensor:
    # shard
    # dim=0 -> column-parallel
    # dim=1 -> row-parallel

    dim = 1 if row_parallel else 0

    if config.world_size > 1:
        x = torch.chunk(x.contiguous(), config.world_size, dim=dim)[config.rank]

    return x


def _shard_and_quantize(
    x: torch.Tensor, config: Config, row_parallel: bool = False
) -> torch.Tensor:

    dim = 1 if row_parallel else 0

    if config.world_size > 1:
        x = torch.chunk(x.contiguous(), config.world_size, dim=dim)[config.rank]

    if config.quantization is not None:
        x = quantize(x, config.quantization)

    return x


def _eval_max_num_kv_pages(
    spec: Spec,
    config: Config,
) -> int:

    available_bytes = get_available_memory(
        devices=config.devices,
        rank=config.rank,
    )
    usable_bytes = available_bytes * config.mem_utilization
    element_size_bytes = torch.empty((), dtype=config.dtype).element_size()
    total_bytes_per_page = (
        element_size_bytes
        * 2
        * config.kv_page_size
        * spec.num_kv_heads
        * spec.dim_head
        * spec.num_layers
    )

    max_num_pages = int(usable_bytes // total_bytes_per_page)
    return max_num_pages


@dataclass
class Spec(SpecBase):

    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    num_vocabs: int

    dim_head: int
    dim_hidden: int
    dim_mlp: int

    rms_norm_eps: float

    rope_factor: float
    rope_high_frequency_factor: float
    rope_low_frequency_factor: float
    rope_theta: float

    @staticmethod
    def from_dict(spec: dict) -> Spec:
        return Spec(
            num_layers=int(spec["num_layers"]),
            num_q_heads=int(spec["num_query_heads"]),
            num_kv_heads=int(spec["num_key_value_heads"]),
            dim_head=int(spec["head_size"]),
            dim_hidden=int(spec["hidden_size"]),
            dim_mlp=int(spec["intermediate_size"]),
            num_vocabs=int(spec["vocab_size"]),
            rms_norm_eps=float(spec["rms_norm_eps"]),
            rope_factor=float(spec["rope"]["factor"]),
            rope_high_frequency_factor=float(spec["rope"]["high_frequency_factor"]),
            rope_low_frequency_factor=float(spec["rope"]["low_frequency_factor"]),
            rope_theta=float(spec["rope"]["theta"]),
        )


@dataclass
class Param(ParamBase):

    spec: Spec
    config: Config

    # linear weights
    proj_down: list[torch.Tensor]
    proj_gate_up: list[torch.Tensor]
    proj_qkv: list[torch.Tensor]
    proj_o: list[torch.Tensor]

    # embedding weight
    embed_token: torch.Tensor

    # norm weights
    norm_attn: list[torch.Tensor]
    norm_mlp: list[torch.Tensor]
    norm_last: torch.Tensor

    @staticmethod
    def from_reader(
        spec: Spec,
        config: Config,
        read: Callable[..., torch.Tensor],
    ) -> Param:

        shardable_dims = {
            "vocab_size": spec.num_vocabs,
            "num_query_heads": spec.num_q_heads,
            "num_key_value_heads": spec.num_kv_heads,
            "intermediate_size": spec.dim_mlp,
        }
        for name, dim in shardable_dims.items():
            if dim % config.world_size != 0:
                raise ValueError(
                    f"Config dimension {name} ({dim}) is not divisible "
                    f"by world_size ({config.world_size})."
                )

        norm_attn = []
        norm_mlp = []
        proj_down = []
        proj_gate_up = []
        proj_qkv = []
        proj_o = []

        # let's not quantize embed weights
        # Don't validate embed vocab size - may differ from config
        embed_token = _shard(
            read("model.embed_tokens.weight"),
            config,
        ).to(config.device)

        for i in range(spec.num_layers):
            prefix = f"model.layers.{i}"

            norm_attn.append(
                read(
                    f"{prefix}.input_layernorm.weight",
                    expected_shape=(spec.dim_hidden,),
                ).to(config.device)
            )
            norm_mlp.append(
                read(
                    f"{prefix}.post_attention_layernorm.weight",
                    expected_shape=(spec.dim_hidden,),
                ).to(config.device)
            )

            # fuse -> shard -> quantize
            proj_qkv.append(
                _shard_and_quantize(
                    torch.cat(
                        [
                            read(
                                f"{prefix}.self_attn.q_proj.weight",
                                expected_shape=(spec.dim_hidden, spec.dim_hidden),
                            ),
                            read(
                                f"{prefix}.self_attn.k_proj.weight",
                                expected_shape=(
                                    spec.num_kv_heads * spec.dim_head,
                                    spec.dim_hidden,
                                ),
                            ),
                            read(
                                f"{prefix}.self_attn.v_proj.weight",
                                expected_shape=(
                                    spec.num_kv_heads * spec.dim_head,
                                    spec.dim_hidden,
                                ),
                            ),
                        ],
                        dim=0,
                    ),
                    config,
                ).to(config.device)
            )

            proj_o.append(
                _shard_and_quantize(
                    read(
                        f"{prefix}.self_attn.o_proj.weight",
                        expected_shape=(spec.dim_hidden, spec.dim_hidden),
                    ),
                    config,
                    row_parallel=True,
                ).to(config.device)
            )

            # fuse -> shard -> quantize
            proj_gate_up.append(
                _shard_and_quantize(
                    torch.cat(
                        [
                            read(
                                f"{prefix}.mlp.gate_proj.weight",
                                expected_shape=(spec.dim_mlp, spec.dim_hidden),
                            ),
                            read(
                                f"{prefix}.mlp.up_proj.weight",
                                expected_shape=(spec.dim_mlp, spec.dim_hidden),
                            ),
                        ],
                        dim=0,
                    ),
                    config,
                ).to(config.device)
            )

            proj_down.append(
                _shard_and_quantize(
                    read(
                        f"{prefix}.mlp.down_proj.weight",
                        expected_shape=(spec.dim_hidden, spec.dim_mlp),
                    ),
                    config,
                    row_parallel=True,
                ).to(config.device)
            )

        norm_last = read("model.norm.weight", expected_shape=(spec.dim_hidden,)).to(
            config.device
        )

        return Param(
            spec=spec,
            config=config,
            proj_down=proj_down,
            proj_gate_up=proj_gate_up,
            proj_qkv=proj_qkv,
            proj_o=proj_o,
            embed_token=embed_token,
            norm_attn=norm_attn,
            norm_mlp=norm_mlp,
            norm_last=norm_last,
        )


@dataclass
class Buffer(BufferBase):

    spec: Spec
    config: Config

    kv_cache: list[torch.Tensor]
    adapter: list[tuple[torch.Tensor, torch.Tensor]]

    decode_attn: ops.BatchDecodeWithPagedKVCacheWrapper
    append_attn: ops.BatchPrefillWithPagedKVCacheWrapper

    @staticmethod
    def from_config(spec: Spec, config: Config) -> Buffer:

        local_num_q_heads = spec.num_q_heads // config.world_size
        local_num_kv_heads = spec.num_kv_heads // config.world_size
        local_q_size = local_num_q_heads * spec.dim_head
        local_kv_size = local_num_kv_heads * spec.dim_head

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=config.device
        )
        decode_attn = ops.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
        append_attn = ops.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

        adapter = [
            (
                torch.zeros(
                    (
                        config.max_num_adapters,
                        config.max_adapter_rank * 3,
                        spec.dim_hidden,
                    ),
                    dtype=config.dtype,
                    device=config.device,
                ),
                torch.zeros(
                    (
                        config.max_num_adapters,
                        spec.dim_head * (local_num_q_heads + local_num_kv_heads * 2),
                        config.max_adapter_rank,
                    ),
                    dtype=config.dtype,
                    device=config.device,
                ),
            )
            for _ in range(spec.num_layers)
        ]

        # update config.max_num_kv_pages
        config.max_num_kv_pages = _eval_max_num_kv_pages(spec, config)

        kv_cache = [
            torch.zeros(
                (
                    config.max_num_kv_pages,
                    2,
                    config.kv_page_size,
                    local_num_kv_heads,
                    spec.dim_head,
                ),
                dtype=config.dtype,
                device=config.device,
            )
            for _ in range(spec.num_layers)
        ]

        return Buffer(
            spec=spec,
            config=config,
            adapter=adapter,
            kv_cache=kv_cache,
            decode_attn=decode_attn,
            append_attn=append_attn,
        )


class ForwardPass:

    def __init__(self, device: torch.device):
        """Initialize the forward pass with attention wrappers."""
        # Create workspace buffer for attention operations
        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )

    def embed_tokens(
        self, param: Param, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Embed token IDs into hidden states."""
        return fun.embedding(token_ids, param.embed_token)

    def lm_head(
        self, param: Param, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Project hidden states to vocabulary logits (weight-tied with embed_tokens)."""
        # Apply final layer norm
        normed = fun.rms_norm(
            hidden_states,
            normalized_shape=[param.spec.dim_hidden],
            weight=param.norm_last,
            eps=param.spec.rms_norm_eps,
        )
        # Project to vocab (weight-tied with embedding)
        return fun.linear(normed, param.embed_token)

    def mlp(
        self, param: Param, hidden_states: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """
        Executes the MLP block for a single layer, including the
        pre-norm and residual connection.
        """
        # --- Calculate local TP sizes ---
        local_mlp_size = param.spec.dim_mlp // param.config.world_size

        # Save input for residual connection
        residual = hidden_states

        # 1. MLP RMSNorm
        normed_input = fun.rms_norm(
            hidden_states,
            normalized_shape=[param.spec.dim_hidden],
            weight=param.norm_mlp[layer_idx],
            eps=param.spec.rms_norm_eps,
        )

        # 2. Gate+Up Projection (Column Parallel)
        gate_up = fun.linear(
            normed_input,
            weight=param.proj_gate_up[layer_idx],
            bias=None,
        )

        # Split gate and up
        gate, up = torch.split(gate_up, [local_mlp_size, local_mlp_size], dim=-1)

        # 3. SiLU activation * gate (SwiGLU)
        hidden = fun.silu(gate) * up

        # 4. Down Projection (Row Parallel)
        down = fun.linear(
            hidden,
            weight=param.proj_down[layer_idx],
            bias=None,
        )
        del hidden, gate, up, gate_up

        # ALL-REDUCE: Sum partial outputs from all ranks (only if TP > 1)
        if param.config.world_size > 1:
            dist.all_reduce(down)

        # 5. Residual Connection
        return residual + down

    def attention(
        self,
        param: Param,
        hidden_states: torch.Tensor,
        layer_idx: int,
        position_ids: torch.Tensor,
        kv_cache_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
        wrapper: Any,
    ) -> torch.Tensor:
        """
        Executes the attention block for a single layer, including the
        pre-norm and residual connection.
        """
        # --- Calculate local TP sizes ---
        local_num_query_heads = param.spec.num_q_heads // param.config.world_size
        local_num_key_value_heads = param.spec.num_kv_heads // param.config.world_size
        local_q_size = local_num_query_heads * param.spec.dim_head
        local_kv_size = local_num_key_value_heads * param.spec.dim_head

        n = hidden_states.size(0)

        # Save input for the first residual connection (replicated)
        residual = hidden_states

        # 1. Input RMSNorm (replicated input -> replicated output)  ------------------> input gather
        normed_input = fun.rms_norm(
            hidden_states,
            normalized_shape=[param.spec.dim_hidden],
            weight=param.norm_attn[layer_idx],
            eps=param.spec.rms_norm_eps,
        )

        # 2. QKV Projection (Column Parallel)
        # Input is replicated, weight is sharded -> output is sharded
        qkv_proj = fun.linear(
            normed_input,
            weight=param.proj_qkv[layer_idx],
            bias=None,
        )

        # q, k, v are all LOCAL shards
        q, k, v = torch.split(
            qkv_proj,
            [
                local_q_size,
                local_kv_size,
                local_kv_size,
            ],
            dim=-1,
        )

        # ------------------------------- cut here -------------------------------

        # 3. Adapter (if any)
        if adapter_subpass is not None:
            adapter_subpass.execute(
                layer_idx,
                normed_input,  # Adapter needs the (replicated) normed input
                q_state=q,
                k_state=k,
                v_state=v,
            )
        del normed_input
        # ------------------------------- cut here -------------------------------

        # 4. Reshape QKV (local shapes) ------------------> input scatter
        q = q.view(n, local_num_query_heads, param.spec.dim_head)
        k = k.view(n, local_num_key_value_heads, param.spec.dim_head)
        v = v.view(n, local_num_key_value_heads, param.spec.dim_head)

        # 5. Apply RoPE (in-place on local shards)
        ops.apply_llama31_rope_pos_ids_inplace(
            q=q,
            k=k,
            pos_ids=position_ids,
            rope_scale=param.spec.rope_factor,
            rope_theta=param.spec.rope_theta,
            low_freq_factor=param.spec.rope_low_frequency_factor,
            high_freq_factor=param.spec.rope_high_frequency_factor,
        )

        # gather where?

        # 6. Append K, V to cache (local shards to local cache)
        # kv_cache_layer is the LOCAL shard of the cache for this layer
        ops.append_paged_kv_cache(
            append_key=k,
            append_value=v,
            batch_indices=batch_indices,
            positions=batch_positions,
            paged_kv_cache=kv_cache_layer,
            kv_indices=kv_page_indices,
            kv_indptr=kv_page_indptr,
            kv_last_page_len=kv_last_page_lens,
            kv_layout="NHD",
        )

        # 7. Compute Attention (on local shards)
        # wrapper was planned with local head counts
        attn_output = wrapper.run(q, kv_cache_layer)
        del q, k, v
        del qkv_proj

        # ------------------------------- cut here -------------------------------

        # attn_output is a local shard
        attn_output = attn_output.reshape(n, -1)

        # 8. Output Projection (Row Parallel)
        # Input is sharded, weight is sharded -> output is partial
        attn_proj = fun.linear(
            attn_output,
            weight=param.proj_o[layer_idx],
            bias=None,
        )
        del attn_output

        # ALL-REDUCE: Sum partial outputs from all ranks (only if TP > 1)
        if param.config.world_size > 1:
            dist.all_reduce(attn_proj)

        # 9. First Residual Connection
        # residual (replicated) + attn_proj (now replicated)
        return residual + attn_proj

    def transform(
        self,
        param: Param,
        # inputs
        input_embeds: torch.Tensor,  # Replicated [n, hidden_size]
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        # kv cache
        kv_cache_at_layer: list[torch.Tensor],  # Each tensor is a LOCAL shard
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        # mask
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
        # subpasses
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:

        # --- Calculate local TP sizes ---
        # <-- These are still needed here for planning the wrapper
        local_num_query_heads = param.spec.num_q_heads // param.config.world_size
        local_num_key_value_heads = param.spec.num_kv_heads // param.config.world_size

        hidden_states = input_embeds
        n, _ = hidden_states.size()

        page_size = int(kv_cache_at_layer[0].shape[2])

        seq_lens = ops.get_seq_lens(
            kv_page_indptr,
            kv_last_page_lens,
            page_size,
        )

        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=seq_lens,
            nnz=n,
        )
        del seq_lens  # No longer needed

        if single_token_inference_mode:
            wrapper = self.wrapper_decode
            wrapper.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_lens,
                num_qo_heads=local_num_query_heads,  # Use local head count
                num_kv_heads=local_num_key_value_heads,  # Use local head count
                head_dim=param.spec.dim_head,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=input_embeds.dtype,
            )
        else:
            wrapper = self.wrapper_append
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_page_indptr,
                paged_kv_indices=kv_page_indices,
                paged_kv_last_page_len=kv_last_page_lens,
                num_qo_heads=local_num_query_heads,  # Use local head count
                num_kv_heads=local_num_key_value_heads,  # Use local head count
                head_dim_qk=param.spec.dim_head,
                page_size=page_size,
                custom_mask=custom_mask,
                q_dta_type=input_embeds.dtype,
            )

        for layer_idx in range(param.spec.num_layers):
            # 1. Attention Block (includes pre-norm and residual)
            hidden_states = self.attention(
                param=param,
                hidden_states=hidden_states,
                layer_idx=layer_idx,
                position_ids=position_ids,
                kv_cache_layer=kv_cache_at_layer[layer_idx],
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                adapter_subpass=adapter_subpass,
                wrapper=wrapper,
            )

            # 2. MLP Block (includes pre-norm and residual)
            hidden_states = self.mlp(
                param=param,
                hidden_states=hidden_states,
                layer_idx=layer_idx,
            )

        # Returns replicated hidden_states
        return hidden_states
