import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from qwen_vit import Qwen2_5_VisionTransformerPretrainedModel

from common import rotate_half, Qwen2_5_VLPreTrainedModel
from l4ma import L4maRmsNorm, L4maMlp, AttentionBuffer, proc_mask, L4maRotaryEmbedding, apply_rotary_pos_emb


#
# class Qwen2_5_VLRotaryEmbedding(nn.Module):
#     def __init__(self, config: Qwen2_5_VLConfig, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings
#
#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
#
#         print(self.rope_type)
#
#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq
#
#     @torch.no_grad()
#     def forward(self, x, position_ids):
#
#         # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for thw grids
#         # So we expand the inv_freq to shape (3, ...)
#         inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
#         position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
#         # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
#         device_type = x.device.type
#         device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()
#
#         # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
#         # cos = cos * self.attention_scaling
#         # sin = sin * self.attention_scaling
#
#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2

    print("original cos: ", cos.shape)

    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    print("new cos: ", cos.shape)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2_5_VLAttention(nn.Module):

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            # position_ids: Optional[torch.LongTensor] = None,
            # past_key_value: Optional[Cache] = None,
            # output_attentions: bool = False,
            # use_cache: bool = False,
            # cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,

            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        # query_states, key_states = apply_multimodal_rotary_pos_emb(
        #     query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        # )

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # print("key_states: ", key_states.shape)
        # print("value_states: ", value_states.shape)

        if buffer is not None:
            buffer.sink(self.layer_idx, buffer_sink_ids, key_states, value_states)
            key_states, value_states = buffer.cache(self.layer_idx, repeat=self.num_key_value_groups)

        # print("akey_states: ", key_states.shape)
        # print("avalue_states: ", value_states.shape)

        # if past_key_value is not None:
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + attention_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2_5_VLAttention(config, layer_idx)

        self.mlp = L4maMlp(config)
        self.input_layernorm = L4maRmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = L4maRmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            # position_ids: Optional[torch.LongTensor] = None,
            # past_key_value: Optional[Tuple[torch.Tensor]] = None,
            # output_attentions: Optional[bool] = False,
            # use_cache: Optional[bool] = False,
            # cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC

            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
            # **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # past_key_value=past_key_value,
            # output_attentions=output_attentions,
            # use_cache=use_cache,
            # cache_position=cache_position,
            position_embeddings=position_embeddings,
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = L4maRmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.rotary_emb_new = L4maRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            inputs_embeds: torch.FloatTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None,
            # past_key_values: Cache | None,
            # cache_position: torch.LongTensor | None,

            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # if past_key_values is None:
        #     past_key_values = DynamicCache()

        attention_mask = proc_mask(attention_mask, inputs_embeds.dtype)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings2 = self.rotary_emb_new(hidden_states, position_ids.max().item() + 1)

        # (bsz, dim)
        # cos_ref, sin_ref = position_embeddings
        cos, sin = position_embeddings2

        dim = cos.shape[-1]
        bsz = hidden_states.shape[0]
        # (3, bsz, dim)
        # print("cos: ", cos.shape)
        # print("sin: ", sin.shape)
        cos_ex = cos[None, None, :, :].expand(3, bsz, -1, -1)
        sin_ex = sin[None, None, :, :].expand(3, bsz, -1, -1)

        # print("position_ids: ", position_ids.max())
        cos = torch.gather(cos_ex, dim=2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, dim))
        sin = torch.gather(sin_ex, dim=2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, dim))

        mrope_section = self.config.rope_scaling["mrope_section"] * 2

        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            1
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            1
        )

        position_embeddings = (cos, sin)

        # assert
        # assert torch.allclose(cos, cos_ref)
        # assert torch.allclose(sin, sin_ref)

        # position_embeddings = (cos, sin)
        # print("cos_ref: ", cos_ref.shape)
        # print("sin_ref: ", sin_ref.shape)
        # print("cos: ", cos.shape)
        # print("sin: ", sin.shape)

        # print("position_embeddings: ", position_embeddings[0][0])
        # print(position_embeddings[0][0].shape)
        # print("posemb: ", posemb[0])
        # print(posemb[0].shape)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                # position_ids=position_ids,
                # past_key_value=past_key_values,
                # cache_position=cache_position,
                position_embeddings=position_embeddings,
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    logits: torch.FloatTensor = None


class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            # cache_position: Optional[torch.LongTensor] = None,
            # past_key_values: Optional[List[torch.FloatTensor]] = None,
            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

        # print all the inputs
        # print("input_ids: ", input_ids)
        # print("attention_mask: ", attention_mask)
        # print("position_ids: ", position_ids)
        # print("past_key_values: ", past_key_values)
        # print("inputs_embeds: ", inputs_embeds)
        # print("use_cache: ", use_cache)
        # print("output_attentions: ", output_attentions)
        # print("output_hidden_states: ", output_hidden_states)
        # print("return_dict: ", return_dict)
        # print("pixel_values: ", pixel_values)
        # print("pixel_values_videos: ", pixel_values_videos)
        # print("image_grid_thw: ", image_grid_thw)
        # print("video_grid_thw: ", video_grid_thw)
        # print("rope_deltas: ", rope_deltas)
        # print("cache_position: ", cache_position)
        # print("second_per_grid_ts: ", second_per_grid_ts)

        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        #     # calculate RoPE index once per generation in the pre-fill stage only
        #     if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
        #         position_ids, rope_deltas = self.get_rope_index(
        #             input_ids,
        #             image_grid_thw,
        #             video_grid_thw,
        #             second_per_grid_ts,
        #             attention_mask,
        #         )
        #         self.rope_deltas = rope_deltas
        #         # print("rope_deltas: ", rope_deltas)
        #     # then use the prev pre-calculated rope-deltas to get the correct position ids
        #     else:
        #         batch_size, seq_length, _ = inputs_embeds.shape
        #         delta = (
        #             (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
        #             if cache_position is not None
        #             else 0
        #         )
        #         # print("cache_position", cache_position)
        #         # print("cache_position[0]", cache_position[0])
        #         # print("delta: ", delta)
        #         position_ids = torch.arange(seq_length, device=inputs_embeds.device)
        #         position_ids = position_ids.view(1, -1).expand(batch_size, -1)
        #         if cache_position is not None:  # otherwise `deltas` is an int `0`
        #             delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
        #         position_ids = position_ids.add(delta)
        #         position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # print("position_ids: ", position_ids)

        # print("position_ids: ", position_ids)
        outputs = self.model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            # past_key_values=past_key_values,
            # cache_position=cache_position,
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        return Qwen2_5_VLCausalLMOutputWithPast(
            logits=logits
        )

    # def prepare_inputs_for_generation(
    #         self,
    #         input_ids,
    #         past_key_values=None,
    #         attention_mask=None,
    #         inputs_embeds=None,
    #         cache_position=None,
    #         position_ids=None,
    #         use_cache=True,
    #         pixel_values=None,
    #         pixel_values_videos=None,
    #         image_grid_thw=None,
    #         video_grid_thw=None,
    #         second_per_grid_ts=None,
    #         **kwargs,
    # ):
    #     # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
    #
    #     # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    #     # Exception 1: when passing input_embeds, input_ids may be missing entries
    #     # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    #     if past_key_values is not None:
    #         if inputs_embeds is not None:  # Exception 1
    #             input_ids = input_ids[:, -cache_position.shape[0]:]
    #         elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
    #             input_ids = input_ids[:, cache_position]
    #
    #     if cache_position[0] != 0:
    #         pixel_values = None
    #         pixel_values_videos = None
    #
    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and cache_position[0] == 0:
    #         model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    #     else:
    #         model_inputs = {"input_ids": input_ids, "inputs_embeds": None}
    #
    #     if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
    #         if model_inputs["inputs_embeds"] is not None:
    #             batch_size, sequence_length, _ = inputs_embeds.shape
    #             device = inputs_embeds.device
    #         else:
    #             batch_size, sequence_length = input_ids.shape
    #             device = input_ids.device
    #
    #         attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
    #             attention_mask,
    #             sequence_length=sequence_length,
    #             target_length=past_key_values.get_max_cache_shape(),
    #             dtype=self.lm_head.weight.dtype,
    #             device=device,
    #             cache_position=cache_position,
    #             batch_size=batch_size,
    #             config=self.config,
    #             past_key_values=past_key_values,
    #         )
    #
    #     model_inputs.update(
    #         {
    #             "position_ids": position_ids,
    #             "past_key_values": past_key_values,
    #             "use_cache": use_cache,
    #             "attention_mask": attention_mask,
    #             "pixel_values": pixel_values,
    #             "pixel_values_videos": pixel_values_videos,
    #             "image_grid_thw": image_grid_thw,
    #             "video_grid_thw": video_grid_thw,
    #             "cache_position": cache_position,
    #             "second_per_grid_ts": second_per_grid_ts,
    #         }
    #     )
    #     return model_inputs
