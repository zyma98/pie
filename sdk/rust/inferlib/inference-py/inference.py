"""Inference interface implementation -- Context and future resources.

This is the most complex module, implementing the Context resource with full
state management and the async future resources (DecodeStepFuture, FlushFuture,
GenerateFuture) that wrap host forward-pass results.
"""

from __future__ import annotations

import math

from wit_world.exports.inference import (
    Context as ContextBase,
    DecodeStepFuture as DecodeStepFutureBase,
    FlushFuture as FlushFutureBase,
    GenerateFuture as GenerateFutureBase,
    SamplerConfig,
    SamplerConfig_Greedy,
    SamplerConfig_Multinomial,
    SamplerConfig_TopP,
    SamplerConfig_TopK,
    SamplerConfig_MinP,
    SamplerConfig_TopKTopP,
    StopConfig,
)
from wit_world.exports.queues import Distribution
from wit_world.imports import inferlet_core_common as _common
from wit_world.imports import runtime as _runtime
from wit_world.imports import tokenize as _tokenize
from wit_world.imports import forward as _forward
from wit_world.imports import inferlet_adapter_common as _adapter
from wit_world.imports import evolve as _zo

from brle import Brle
from kv_page import KvPageManager
from formatter import ChatFormatter


def _greedy_argmax(ids: list[int], probs: list[float]) -> int:
    if not ids:
        return 0
    max_idx = 0
    max_val = probs[0]
    for i in range(1, len(probs)):
        if probs[i] > max_val:
            max_val = probs[i]
            max_idx = i
    return ids[max_idx]


class _ContextInner:
    """Internal context state, separate from the WIT resource wrapper."""

    def __init__(self, host_model):
        self._host_model = host_model
        self._host_queue = host_model.create_queue()
        self._tokenizer = _tokenize.get_tokenizer(host_model)
        self._kv_page_size = host_model.get_kv_page_size()
        self._formatter = ChatFormatter(host_model.get_prompt_template())

        self._token_ids: list[int] = []
        self._pending_tokens: list[int] = []

        self._token_mask_current = Brle.new(0)
        self._token_mask_pending: list[Brle] = []

        self._position_ids: list[int] = []

        self._kv_manager = KvPageManager(self._host_queue, self._kv_page_size)

        self._adapter_ptr: int | None = None
        self._adapter_random_seed: int | None = None

        self._begin_of_sequence = True

    @classmethod
    def from_imported_state(
        cls,
        host_model,
        kv_page_ptrs: list[int],
        prefix_tokens: list[int],
        kv_page_last_len: int,
    ) -> _ContextInner:
        ctx = cls(host_model)
        ctx._token_ids = list(prefix_tokens)
        ctx._position_ids = list(range(len(prefix_tokens)))
        ctx._kv_manager.import_pages_from_state(kv_page_ptrs, kv_page_last_len)
        ctx._token_mask_current = Brle.new(len(prefix_tokens))
        ctx._begin_of_sequence = False
        return ctx

    def fill(self, text: str) -> None:
        tokens = list(self._tokenizer.tokenize(text))
        self.fill_tokens(tokens)

    def fill_tokens(self, tokens: list[int]) -> None:
        self._pending_tokens.extend(tokens)
        for _ in range(len(tokens)):
            self._token_mask_current.append(False)
            self._token_mask_pending.append(self._token_mask_current.clone())
        self._begin_of_sequence = False

    def fill_token(self, token_id: int) -> None:
        self._pending_tokens.append(token_id)
        self._token_mask_current.append(False)
        self._token_mask_pending.append(self._token_mask_current.clone())
        self._begin_of_sequence = False

    def fill_system(self, text: str) -> None:
        self._formatter.add_system(text)
        self._flush_chat_messages(False)

    def fill_user(self, text: str) -> None:
        self._formatter.add_user(text)
        self._flush_chat_messages(True)

    def fill_user_only(self, text: str) -> None:
        self._formatter.add_user(text)
        self._flush_chat_messages(False)

    def fill_assistant(self, text: str) -> None:
        self._formatter.add_assistant(text)
        self._flush_chat_messages(False)

    def _flush_chat_messages(self, add_generation_prompt: bool) -> None:
        if self._formatter.has_messages():
            rendered = self._formatter.render(add_generation_prompt, self._begin_of_sequence)
            self._begin_of_sequence = False
            self._formatter.clear()
            self.fill(rendered)

    def mask_tokens(self, indices: list[int], mask: bool) -> None:
        self._token_mask_current.mask(indices, mask)

    def mask_token_range(self, start: int, end: int, mask: bool) -> None:
        self._token_mask_current.mask_range(start, end, mask)

    def mask_token(self, index: int, mask: bool) -> None:
        self._token_mask_current.mask([index], mask)

    def drop_masked_kv_pages(self) -> None:
        num_committed_pages = len(self._token_ids) // self._kv_page_size

        for i in range(num_committed_pages - 1, -1, -1):
            page_start = i * self._kv_page_size
            page_end = (i + 1) * self._kv_page_size

            if self._token_mask_current.is_range_all_value(page_start, page_end, True):
                self._kv_manager.remove_page_at(i)
                del self._token_ids[page_start:page_end]
                del self._position_ids[page_start:page_end]
                self._token_mask_current.remove_range(page_start, page_end)
                for m in self._token_mask_pending:
                    m.remove_range(page_start, page_end)

        self._kv_manager.recalculate_last_page_len(len(self._token_ids))

    def set_adapter(self, adapter_ptr: int) -> None:
        self._adapter_ptr = adapter_ptr

    def remove_adapter(self) -> None:
        self._adapter_ptr = None

    def set_adapter_random_seed(self, seed: int) -> None:
        self._adapter_random_seed = seed

    def flush(self) -> None:
        if not self._pending_tokens:
            return

        pending_token_ids = self._pending_tokens[:]
        self._pending_tokens.clear()

        pending_masks = self._token_mask_pending[:]
        self._token_mask_pending.clear()
        mask_buffers = [m.buffer for m in pending_masks]

        last_pos = (self._position_ids[-1] + 1) if self._position_ids else 0
        position_ids = list(range(last_pos, last_pos + len(pending_token_ids)))

        self._kv_manager.grow(len(pending_token_ids))

        fp = _forward.create_forward_pass(self._host_queue)
        _forward.input_tokens(fp, pending_token_ids, position_ids)
        _forward.kv_cache(fp, self._kv_manager.page_ptrs, self._kv_manager.last_page_len)
        _forward.attention_mask(fp, mask_buffers)

        host_result = fp.execute()
        if host_result is not None:
            pollable = host_result.pollable()
            pollable.block()

        self._token_ids.extend(pending_token_ids)
        self._position_ids.extend(position_ids)

    def _submit_decode_step(self, sampler: SamplerConfig):
        """Submit a decode step without blocking. Returns (host_result, pending_token_ids, position_ids)."""
        assert self._pending_tokens, "Must have at least one seed token"

        pending_token_ids = self._pending_tokens[:]
        self._pending_tokens.clear()

        pending_masks = self._token_mask_pending[:]
        self._token_mask_pending.clear()
        mask_buffers = [m.buffer for m in pending_masks]

        last_pos = (self._position_ids[-1] + 1) if self._position_ids else 0
        position_ids = list(range(last_pos, last_pos + len(pending_token_ids)))

        self._kv_manager.grow(len(pending_token_ids))

        fp = _forward.create_forward_pass(self._host_queue)

        if self._adapter_ptr is not None:
            _adapter.set_adapter(fp, self._adapter_ptr)
            if self._adapter_random_seed is not None:
                _zo.set_adapter_seed(fp, self._adapter_random_seed)

        _forward.input_tokens(fp, pending_token_ids, position_ids)
        _forward.kv_cache(fp, self._kv_manager.page_ptrs, self._kv_manager.last_page_len)
        _forward.attention_mask(fp, mask_buffers)

        output_idx = [len(pending_token_ids) - 1]
        _apply_sampler_to_fp(fp, output_idx, sampler)

        host_result = fp.execute()
        assert host_result is not None, "Forward pass returned no result"
        return host_result, pending_token_ids, position_ids

    def _commit_decode_step(self, pending_token_ids: list[int], position_ids: list[int]) -> None:
        self._token_ids.extend(pending_token_ids)
        self._position_ids.extend(position_ids)

    def decode_step(self, sampler: SamplerConfig) -> int:
        host_result, pending_token_ids, position_ids = self._submit_decode_step(sampler)

        pollable = host_result.pollable()
        pollable.block()
        tokens = host_result.get_tokens()
        sampled = tokens[0]

        self._commit_decode_step(pending_token_ids, position_ids)
        return sampled

    def decode_step_dist(self, temperature: float, top_k: int | None) -> tuple[list[int], list[float]]:
        assert self._pending_tokens, "Must have at least one seed token"

        pending_token_ids = self._pending_tokens[:]
        self._pending_tokens.clear()

        pending_masks = self._token_mask_pending[:]
        self._token_mask_pending.clear()
        mask_buffers = [m.buffer for m in pending_masks]

        last_pos = (self._position_ids[-1] + 1) if self._position_ids else 0
        position_ids = list(range(last_pos, last_pos + len(pending_token_ids)))

        self._kv_manager.grow(len(pending_token_ids))

        fp = _forward.create_forward_pass(self._host_queue)

        if self._adapter_ptr is not None:
            _adapter.set_adapter(fp, self._adapter_ptr)
            if self._adapter_random_seed is not None:
                _zo.set_adapter_seed(fp, self._adapter_random_seed)

        _forward.input_tokens(fp, pending_token_ids, position_ids)
        _forward.kv_cache(fp, self._kv_manager.page_ptrs, self._kv_manager.last_page_len)
        _forward.attention_mask(fp, mask_buffers)

        output_idx = [len(pending_token_ids) - 1]
        _forward.output_distributions(fp, output_idx, temperature, top_k)

        host_result = fp.execute()
        assert host_result is not None

        pollable = host_result.pollable()
        pollable.block()

        raw_dists = host_result.get_distributions()
        assert raw_dists is not None and len(raw_dists) > 0

        ids, probs = raw_dists[0]

        self._token_ids.extend(pending_token_ids)
        self._position_ids.extend(position_ids)

        return list(ids), list(probs)

    def generate(self, sampler: SamplerConfig, stop_config: StopConfig) -> str:
        generated = []
        while True:
            token = self.decode_step(sampler)
            self.fill_token(token)
            generated.append(token)

            if len(generated) >= stop_config.max_tokens:
                break
            if any(
                len(seq) > 0
                and len(generated) >= len(seq)
                and generated[-len(seq):] == list(seq)
                for seq in stop_config.eos_sequences
            ):
                break

        return self._tokenizer.detokenize(generated)

    def generate_with_beam(self, stop_config: StopConfig, beam_size: int) -> str:
        beams: list[tuple[_ContextInner, list[int], float]] = [
            (self.fork(), [], 0.0)
        ]

        while True:
            for beam, generated, _score in beams:
                if len(generated) >= stop_config.max_tokens:
                    result = self._tokenizer.detokenize(generated)
                    self._adopt_state(beam)
                    return result
                if any(
                    len(seq) > 0
                    and len(generated) >= len(seq)
                    and generated[-len(seq):] == list(seq)
                    for seq in stop_config.eos_sequences
                ):
                    result = self._tokenizer.detokenize(generated)
                    self._adopt_state(beam)
                    return result

            all_dists = []
            for beam, _gen, _score in beams:
                dist = beam.decode_step_dist(1.0, None)
                all_dists.append(dist)

            next_beams: list[tuple[_ContextInner, list[int], float]] = []
            for (beam, generated, score), (ids, probs) in zip(beams, all_dists):
                expand_count = min(beam_size, len(ids))
                for j in range(expand_count):
                    if probs[j] <= 0:
                        continue
                    next_beam = beam.fork()
                    next_beam.fill_token(ids[j])
                    next_generated = generated + [ids[j]]
                    next_score = score + math.log(probs[j])
                    next_beams.append((next_beam, next_generated, next_score))

            next_beams.sort(key=lambda x: x[2], reverse=True)
            beams = next_beams[:beam_size]

    def verify_draft(self, draft_tokens: list[int], draft_pos_ids: list[int]) -> list[int]:
        assert self._pending_tokens, "Must have at least one seed token"

        pending_token_ids = self._pending_tokens[:]
        self._pending_tokens.clear()
        self._token_mask_pending.clear()

        batch_tokens = pending_token_ids + list(draft_tokens)
        pending_len = len(pending_token_ids)
        pos_offset = (self._position_ids[-1] + 1) if self._position_ids else 0

        batch_positions = list(range(pos_offset, pos_offset + pending_len))
        batch_positions.extend(
            pos_offset + pending_len - 1 + pos for pos in draft_pos_ids
        )

        # Build attention masks
        batch_masks = []
        pending_brle = Brle.new(pos_offset)

        for _ in range(pending_len):
            pending_brle.append(False)
            batch_masks.append(pending_brle.buffer)

        draft_mask = [True] * len(draft_tokens)
        predecessors: list[tuple[int, int]] = []  # (draft_mask_idx, pos)

        for batch_idx in range(pending_len, len(batch_tokens)):
            draft_mask_idx = batch_idx - pending_len
            pos = batch_positions[batch_idx]

            while predecessors and predecessors[-1][1] != pos - 1:
                prev_idx = predecessors[-1][0]
                draft_mask[prev_idx] = True
                predecessors.pop()

            draft_mask[draft_mask_idx] = False
            predecessors.append((draft_mask_idx, pos))

            brle = pending_brle.clone()
            brle.extend(Brle.from_bools(draft_mask[: draft_mask_idx + 1]))
            batch_masks.append(brle.buffer)

        self._kv_manager.grow(len(batch_tokens))

        out_start = pending_len - 1
        out_indices = list(range(out_start, len(batch_tokens)))

        fp = _forward.create_forward_pass(self._host_queue)

        if self._adapter_ptr is not None:
            _adapter.set_adapter(fp, self._adapter_ptr)
            if self._adapter_random_seed is not None:
                _zo.set_adapter_seed(fp, self._adapter_random_seed)

        _forward.input_tokens(fp, batch_tokens, batch_positions)
        _forward.kv_cache(fp, self._kv_manager.page_ptrs, self._kv_manager.last_page_len)
        _forward.attention_mask(fp, batch_masks)
        _forward.output_distributions(fp, out_indices, 0.0, None)

        host_result = fp.execute()
        assert host_result is not None

        pollable = host_result.pollable()
        pollable.block()

        raw_dists = host_result.get_distributions()
        assert raw_dists is not None

        # Greedy verification
        accepted = []
        first_ids, first_probs = raw_dists[0]
        first_token = _greedy_argmax(list(first_ids), list(first_probs))
        accepted.append(first_token)

        draft_token_idx = 0
        while draft_token_idx < len(draft_tokens):
            last_accepted = accepted[-1]
            draft_token = draft_tokens[draft_token_idx]

            if last_accepted == draft_token:
                next_ids, next_probs = raw_dists[draft_token_idx + 1]
                next_token = _greedy_argmax(list(next_ids), list(next_probs))
                accepted.append(next_token)

                has_child = (
                    draft_token_idx + 1 < len(draft_tokens)
                    and draft_pos_ids[draft_token_idx] + 1 == draft_pos_ids[draft_token_idx + 1]
                )
                if has_child:
                    draft_token_idx += 1
                else:
                    break
            else:
                next_sibling_idx = None
                cur_depth = draft_pos_ids[draft_token_idx]
                for idx in range(draft_token_idx + 1, len(draft_tokens)):
                    if draft_pos_ids[idx] < cur_depth:
                        break
                    if draft_pos_ids[idx] == cur_depth:
                        next_sibling_idx = idx
                        break

                if next_sibling_idx is not None:
                    draft_token_idx = next_sibling_idx
                else:
                    break

        self._kv_manager.shrink(len(draft_tokens))

        self._position_ids.extend(batch_positions[:pending_len])
        self._token_ids.extend(pending_token_ids)

        self.fill_tokens(accepted[:])
        return accepted

    def fork(self) -> _ContextInner:
        model_name = self._host_model.get_name()
        host_model = _runtime.get_model(model_name)
        assert host_model is not None

        forked = _ContextInner(host_model)
        forked._begin_of_sequence = self._begin_of_sequence
        forked._adapter_ptr = self._adapter_ptr
        forked._adapter_random_seed = self._adapter_random_seed

        if (
            self._kv_manager.last_page_len == self._kv_page_size
            and self._pending_tokens
        ):
            forked._token_ids = self._token_ids[:]
            forked._pending_tokens = self._pending_tokens[:]
            forked._kv_manager.import_pages_from_state(
                self._kv_manager.page_ptrs[:], self._kv_manager.last_page_len
            )
            forked._position_ids = self._position_ids[:]
            forked._token_mask_pending = [m.clone() for m in self._token_mask_pending]
            forked._token_mask_current = self._token_mask_current.clone()
        else:
            kept_kv_page_len = max(0, self._kv_manager.page_count - 1)
            kept_tokens_len = kept_kv_page_len * self._kv_page_size

            forked._token_ids = self._token_ids[:kept_tokens_len]
            forked._position_ids = self._position_ids[:kept_tokens_len]

            forked_kv_ptrs = self._kv_manager.page_ptrs[:kept_kv_page_len]
            forked_last_len = self._kv_page_size if kept_kv_page_len > 0 else 0
            forked._kv_manager.import_pages_from_state(forked_kv_ptrs, forked_last_len)

            forked._pending_tokens = (
                self._token_ids[kept_tokens_len:] + self._pending_tokens[:]
            )

            mask_builder = self._token_mask_current.clone()
            parent_total = len(self._token_ids) + len(self._pending_tokens)
            mask_builder.remove_range(kept_tokens_len, parent_total)

            forked._token_mask_pending = []
            for _ in range(len(forked._pending_tokens)):
                mask_builder.append(False)
                forked._token_mask_pending.append(mask_builder.clone())

            forked._token_mask_current = mask_builder

        return forked

    def _adopt_state(self, other: _ContextInner) -> None:
        self._token_ids = other._token_ids[:]
        self._pending_tokens = other._pending_tokens[:]
        self._position_ids = other._position_ids[:]
        self._kv_manager.import_pages_from_state(
            other._kv_manager.page_ptrs[:], other._kv_manager.last_page_len
        )

    def get_text(self) -> str:
        return self._tokenizer.detokenize(self._token_ids)

    def get_token_ids(self) -> list[int]:
        return self._token_ids[:]

    def get_kv_page_ptrs(self) -> list[int]:
        return self._kv_manager.page_ptrs[:]

    def get_kv_page_last_len(self) -> int:
        return self._kv_manager.last_page_len


def _apply_sampler_to_fp(fp, output_idx: list[int], sampler: SamplerConfig) -> None:
    if isinstance(sampler, SamplerConfig_Greedy):
        _forward.output_tokens(fp, output_idx, 0.0)
    elif isinstance(sampler, SamplerConfig_Multinomial):
        _forward.output_tokens(fp, output_idx, sampler.value)
    elif isinstance(sampler, SamplerConfig_TopP):
        temp, top_p = sampler.value
        _forward.output_tokens_top_p(fp, output_idx, temp, top_p)
    elif isinstance(sampler, SamplerConfig_TopK):
        temp, top_k = sampler.value
        _forward.output_tokens_top_k(fp, output_idx, temp, top_k)
    elif isinstance(sampler, SamplerConfig_MinP):
        temp, min_p = sampler.value
        _forward.output_tokens_min_p(fp, output_idx, temp, min_p)
    elif isinstance(sampler, SamplerConfig_TopKTopP):
        temp, top_k, top_p = sampler.value
        _forward.output_tokens_top_k_top_p(fp, output_idx, temp, top_k, top_p)


class Context(ContextBase):
    def __init__(self, model):
        model_impl = model
        host_model = _runtime.get_model(model_impl.get_name())
        assert host_model is not None
        self._inner = _ContextInner(host_model)

    @classmethod
    def from_imported_state(cls, model, kv_page_ptrs, prefix_tokens, kv_page_last_len):
        host_model = _runtime.get_model(model.get_name())
        assert host_model is not None
        instance = cls.__new__(cls)
        instance._inner = _ContextInner.from_imported_state(
            host_model, list(kv_page_ptrs), list(prefix_tokens), kv_page_last_len
        )
        return instance

    def fill(self, text: str) -> None:
        self._inner.fill(text)

    def fill_tokens(self, token_ids: list[int]) -> None:
        self._inner.fill_tokens(list(token_ids))

    def fill_token(self, token_id: int) -> None:
        self._inner.fill_token(token_id)

    def fill_system(self, text: str) -> None:
        self._inner.fill_system(text)

    def fill_user(self, text: str) -> None:
        self._inner.fill_user(text)

    def fill_user_only(self, text: str) -> None:
        self._inner.fill_user_only(text)

    def fill_assistant(self, text: str) -> None:
        self._inner.fill_assistant(text)

    def mask_tokens(self, indices: list[int], mask: bool) -> None:
        self._inner.mask_tokens(list(indices), mask)

    def mask_token_range(self, start: int, end: int, mask: bool) -> None:
        self._inner.mask_token_range(start, end, mask)

    def mask_token(self, index: int, mask: bool) -> None:
        self._inner.mask_token(index, mask)

    def drop_masked_kv_pages(self) -> None:
        self._inner.drop_masked_kv_pages()

    def set_adapter(self, adapter_ptr: int) -> None:
        self._inner.set_adapter(adapter_ptr)

    def remove_adapter(self) -> None:
        self._inner.remove_adapter()

    def set_adapter_random_seed(self, seed: int) -> None:
        self._inner.set_adapter_random_seed(seed)

    def flush(self) -> None:
        self._inner.flush()

    def decode_step(self, sampler: SamplerConfig) -> int:
        return self._inner.decode_step(sampler)

    def decode_step_dist(self, temperature: float, top_k: int | None) -> Distribution:
        ids, probs = self._inner.decode_step_dist(temperature, top_k)
        return Distribution(ids=ids, probs=probs)

    def generate(self, sampler: SamplerConfig, stop_config: StopConfig) -> str:
        return self._inner.generate(sampler, stop_config)

    def generate_with_beam(self, stop_config: StopConfig, beam_size: int) -> str:
        return self._inner.generate_with_beam(stop_config, beam_size)

    def verify_draft(self, draft_tokens: list[int], draft_pos_ids: list[int]) -> list[int]:
        return self._inner.verify_draft(list(draft_tokens), list(draft_pos_ids))

    def fork(self):
        forked_inner = self._inner.fork()
        new_ctx = Context.__new__(Context)
        new_ctx._inner = forked_inner
        return new_ctx

    def get_text(self) -> str:
        return self._inner.get_text()

    def get_token_ids(self) -> list[int]:
        return self._inner.get_token_ids()

    def get_kv_page_ptrs(self) -> list[int]:
        return self._inner.get_kv_page_ptrs()

    def get_kv_page_last_len(self) -> int:
        return self._inner.get_kv_page_last_len()

    def decode_step_async(self, sampler: SamplerConfig):
        host_result, pending_token_ids, position_ids = self._inner._submit_decode_step(sampler)
        return DecodeStepFuture(self._inner, host_result, pending_token_ids, position_ids)

    def flush_async(self):
        if not self._inner._pending_tokens:
            return None

        pending_token_ids = self._inner._pending_tokens[:]
        self._inner._pending_tokens.clear()

        pending_masks = self._inner._token_mask_pending[:]
        self._inner._token_mask_pending.clear()
        mask_buffers = [m.buffer for m in pending_masks]

        last_pos = (self._inner._position_ids[-1] + 1) if self._inner._position_ids else 0
        position_ids = list(range(last_pos, last_pos + len(pending_token_ids)))

        self._inner._kv_manager.grow(len(pending_token_ids))

        fp = _forward.create_forward_pass(self._inner._host_queue)
        _forward.input_tokens(fp, pending_token_ids, position_ids)
        _forward.kv_cache(fp, self._inner._kv_manager.page_ptrs, self._inner._kv_manager.last_page_len)
        _forward.attention_mask(fp, mask_buffers)

        host_result = fp.execute()

        self._inner._token_ids.extend(pending_token_ids)
        self._inner._position_ids.extend(position_ids)

        return FlushFuture(host_result)

    def generate_async(self, sampler: SamplerConfig, stop_config: StopConfig):
        return GenerateFuture(self._inner, sampler, stop_config)


class DecodeStepFuture(DecodeStepFutureBase):
    def __init__(self, ctx: _ContextInner, host_result, pending_token_ids, position_ids):
        self._ctx = ctx
        self._host_result = host_result
        self._pending_token_ids = pending_token_ids
        self._position_ids = position_ids

    def pollable(self):
        return self._host_result.pollable()

    def get(self) -> int | None:
        if self._host_result is None:
            return None
        tokens = self._host_result.get_tokens()
        if tokens is None:
            return None
        sampled = tokens[0]
        self._ctx._commit_decode_step(self._pending_token_ids, self._position_ids)
        return sampled


class FlushFuture(FlushFutureBase):
    def __init__(self, host_result):
        self._host_result = host_result

    def pollable(self):
        if self._host_result is not None:
            return self._host_result.pollable()
        raise RuntimeError("Flush future has no pending result")

    def is_ready(self) -> bool:
        if self._host_result is not None:
            return self._host_result.pollable().ready()
        return True


# GenerateFuture state machine phases
_PHASE_READY = 0
_PHASE_PENDING = 1
_PHASE_DONE = 2


class GenerateFuture(GenerateFutureBase):
    def __init__(self, ctx: _ContextInner, sampler: SamplerConfig, stop_config: StopConfig):
        self._ctx = ctx
        self._sampler = sampler
        self._stop_config = stop_config
        self._generated: list[int] = []
        self._phase = _PHASE_READY
        self._host_result = None
        self._pending_token_ids: list[int] = []
        self._pending_position_ids: list[int] = []

    def pollable(self):
        if self._phase == _PHASE_READY:
            host_result, ptids, pids = self._ctx._submit_decode_step(self._sampler)
            self._host_result = host_result
            self._pending_token_ids = ptids
            self._pending_position_ids = pids
            self._phase = _PHASE_PENDING
            return host_result.pollable()
        elif self._phase == _PHASE_PENDING:
            return self._host_result.pollable()
        else:
            raise RuntimeError("pollable() called on completed generate future")

    def get(self) -> str | None:
        if self._phase != _PHASE_PENDING:
            return None

        tokens = self._host_result.get_tokens()
        assert tokens is not None, "Decode step produced no token"
        token = tokens[0]

        self._ctx._commit_decode_step(self._pending_token_ids, self._pending_position_ids)
        self._ctx.fill_token(token)

        self._generated.append(token)

        should_stop = len(self._generated) >= self._stop_config.max_tokens
        if not should_stop:
            should_stop = any(
                len(seq) > 0
                and len(self._generated) >= len(seq)
                and self._generated[-len(seq):] == list(seq)
                for seq in self._stop_config.eos_sequences
            )

        if should_stop:
            result = self._ctx._tokenizer.detokenize(self._generated)
            self._phase = _PHASE_DONE
            return result
        else:
            self._phase = _PHASE_READY
            return None
