"""Queues interface implementation -- Queue and ForwardPass resources."""

from wit_world.exports.queues import (
    Queue as QueueBase,
    ForwardPass as ForwardPassBase,
    ForwardPassResult,
    Distribution,
    Priority,
)
from wit_world.imports import inferlet_core_common as _common
from wit_world.imports import runtime as _runtime
from wit_world.imports import forward as _forward
from wit_world.imports import inferlet_adapter_common as _adapter
from wit_world.imports import evolve as _zo
from wit_world.imports import image as _image

KV_PAGE_TYPE = 0
EMBED_TYPE = 1
ADAPTER_TYPE = 2


class ForwardPass(ForwardPassBase):
    def __init__(self, host_fp):
        self._inner = host_fp

    def input_tokens(self, tokens: list[int], positions: list[int]) -> None:
        _forward.input_tokens(self._inner, tokens, positions)

    def input_embed_ptrs(self, embed_ptrs: list[int], positions: list[int]) -> None:
        _forward.input_embeddings(self._inner, embed_ptrs, positions)

    def kv_cache(self, kv_page_ptrs: list[int], last_kv_page_len: int) -> None:
        _forward.kv_cache(self._inner, kv_page_ptrs, last_kv_page_len)

    def attention_mask(self, mask: list[list[int]]) -> None:
        _forward.attention_mask(self._inner, mask)

    def set_adapter(self, adapter_ptr: int) -> None:
        _adapter.set_adapter(self._inner, adapter_ptr)

    def set_adapter_seed(self, seed: int) -> None:
        _zo.set_adapter_seed(self._inner, seed)

    def output_distributions(self, indices: list[int], temperature: float, top_k: int | None) -> None:
        _forward.output_distributions(self._inner, indices, temperature, top_k)

    def output_tokens(self, indices: list[int], temperature: float) -> None:
        _forward.output_tokens(self._inner, indices, temperature)

    def output_tokens_top_p(self, indices: list[int], temperature: float, top_p: float) -> None:
        _forward.output_tokens_top_p(self._inner, indices, temperature, top_p)

    def output_tokens_top_k(self, indices: list[int], temperature: float, top_k: int) -> None:
        _forward.output_tokens_top_k(self._inner, indices, temperature, top_k)

    def output_tokens_min_p(self, indices: list[int], temperature: float, min_p: float) -> None:
        _forward.output_tokens_min_p(self._inner, indices, temperature, min_p)

    def output_tokens_top_k_top_p(self, indices: list[int], temperature: float, top_k: int, top_p: float) -> None:
        _forward.output_tokens_top_k_top_p(self._inner, indices, temperature, top_k, top_p)

    def output_embed_ptrs(self, embed_ptrs: list[int], indices: list[int]) -> None:
        _forward.output_embeddings(self._inner, embed_ptrs, indices)

    def execute(self) -> ForwardPassResult:
        host_result = self._inner.execute()
        if host_result is None:
            return ForwardPassResult(distributions=None, tokens=None)

        pollable = host_result.pollable()
        pollable.block()

        tokens = host_result.get_tokens()
        raw_dists = host_result.get_distributions()

        distributions = None
        if raw_dists is not None:
            distributions = [
                Distribution(ids=list(ids), probs=list(probs))
                for ids, probs in raw_dists
            ]

        return ForwardPassResult(
            distributions=distributions,
            tokens=list(tokens) if tokens is not None else None,
        )


class Queue(QueueBase):
    def __init__(self, host_queue, service_id: int):
        self._inner = host_queue
        self._service_id = service_id

    @classmethod
    def from_model_name(cls, model_name: str):
        host_model = _runtime.get_model(model_name)
        if host_model is None:
            raise ValueError(f"Model '{model_name}' not found")
        host_queue = host_model.create_queue()
        service_id = host_model.get_service_id()
        return cls(host_queue, service_id)

    def get_service_id(self) -> int:
        return self._service_id

    def synchronize(self) -> bool:
        result = self._inner.synchronize()
        while True:
            pollable = result.pollable()
            pollable.block()
            value = result.get()
            if value is not None:
                return value

    def set_priority(self, priority: Priority) -> None:
        host_priority = {
            Priority.LOW: _common.Priority.LOW,
            Priority.NORMAL: _common.Priority.NORMAL,
            Priority.HIGH: _common.Priority.HIGH,
        }[priority]
        self._inner.set_priority(host_priority)

    def debug_query(self, query: str) -> str:
        result = self._inner.debug_query(query)
        while True:
            pollable = result.pollable()
            pollable.block()
            value = result.get()
            if value is not None:
                return value

    def allocate_kv_pages(self, count: int) -> list[int]:
        return list(_common.allocate_resources(self._inner, KV_PAGE_TYPE, count))

    def deallocate_kv_pages(self, ptrs: list[int]) -> None:
        _common.deallocate_resources(self._inner, KV_PAGE_TYPE, ptrs)

    def export_kv_pages(self, ptrs: list[int], name: str) -> None:
        _common.export_resources(self._inner, KV_PAGE_TYPE, ptrs, name)

    def import_kv_pages(self, name: str) -> list[int]:
        return list(_common.import_resources(self._inner, KV_PAGE_TYPE, name))

    def get_all_exported_kv_pages(self) -> list[tuple[str, int]]:
        return list(_common.get_all_exported_resources(self._inner, KV_PAGE_TYPE))

    def release_exported_kv_pages(self, name: str) -> None:
        _common.release_exported_resources(self._inner, KV_PAGE_TYPE, name)

    def allocate_embeds(self, count: int) -> list[int]:
        return list(_common.allocate_resources(self._inner, EMBED_TYPE, count))

    def deallocate_embeds(self, ptrs: list[int]) -> None:
        _common.deallocate_resources(self._inner, EMBED_TYPE, ptrs)

    def export_embeds(self, ptrs: list[int], name: str) -> None:
        _common.export_resources(self._inner, EMBED_TYPE, ptrs, name)

    def import_embeds(self, name: str) -> list[int]:
        return list(_common.import_resources(self._inner, EMBED_TYPE, name))

    def get_all_exported_embeds(self) -> list[tuple[str, int]]:
        return list(_common.get_all_exported_resources(self._inner, EMBED_TYPE))

    def release_exported_embeds(self, name: str) -> None:
        _common.release_exported_resources(self._inner, EMBED_TYPE, name)

    def allocate_adapter(self) -> int:
        ptrs = _common.allocate_resources(self._inner, ADAPTER_TYPE, 1)
        return ptrs[0]

    def deallocate_adapter(self, ptr: int) -> None:
        _common.deallocate_resources(self._inner, ADAPTER_TYPE, [ptr])

    def export_adapter(self, ptr: int, name: str) -> None:
        _common.export_resources(self._inner, ADAPTER_TYPE, [ptr], name)

    def import_adapter(self, name: str) -> int:
        ptrs = _common.import_resources(self._inner, ADAPTER_TYPE, name)
        return ptrs[0]

    def get_all_exported_adapters(self) -> list[str]:
        return [name for name, _ in _common.get_all_exported_resources(self._inner, ADAPTER_TYPE)]

    def release_exported_adapter(self, name: str) -> None:
        _common.release_exported_resources(self._inner, ADAPTER_TYPE, name)

    def upload_adapter(self, adapter_ptr: int, name: str, data: bytes) -> None:
        blob = _common.Blob(data)
        _adapter.upload_adapter(self._inner, adapter_ptr, name, blob)

    def download_adapter(self, adapter_ptr: int, name: str) -> None:
        _adapter.download_adapter(self._inner, adapter_ptr, name)

    def initialize_adapter(
        self,
        adapter_ptr: int,
        rank: int,
        alpha: float,
        population_size: int,
        mu_fraction: float,
        initial_sigma: float,
    ) -> None:
        _zo.initialize_adapter(
            self._inner, adapter_ptr, rank, alpha,
            population_size, mu_fraction, initial_sigma,
        )

    def update_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ) -> None:
        _zo.update_adapter(self._inner, adapter_ptr, scores, seeds, max_sigma)

    def embed_image(self, embed_ptrs: list[int], image_data: bytes, position_offset: int) -> None:
        _image.embed_image(self._inner, embed_ptrs, image_data, position_offset)

    def calculate_embed_size(self, image_width: int, image_height: int) -> int:
        return _image.calculate_embed_size(self._inner, image_width, image_height)

    def create_forward_pass(self):
        host_fp = _forward.create_forward_pass(self._inner)
        return ForwardPass(host_fp)
