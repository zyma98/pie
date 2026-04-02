"""KV page management for the inference-py component.

Adapted from sdk/python/src/inferlet/kv_page.py to use host API imports
directly (inferlet:core/common for resource allocation).
"""

from __future__ import annotations

import math

from wit_world.imports.inferlet_core_common import deallocate_resources, allocate_resources

KV_PAGE_TYPE = 0


class KvPage:
    """Reference-counted KV cache page wrapping a host pointer."""

    def __init__(self, host_queue, ptr: int) -> None:
        self._host_queue = host_queue
        self._ptr = ptr
        self._ref_count = 1
        self._released = False

    @property
    def ptr(self) -> int:
        return self._ptr

    def ref(self) -> None:
        self._ref_count += 1

    def release(self) -> None:
        if self._released:
            return
        self._ref_count -= 1
        if self._ref_count <= 0:
            deallocate_resources(self._host_queue, KV_PAGE_TYPE, [self._ptr])
            self._released = True


class KvPageManager:
    """Manages KV cache pages for a context."""

    def __init__(self, host_queue, page_size: int) -> None:
        self._host_queue = host_queue
        self._page_size = page_size
        self._pages: list[KvPage] = []
        self._last_page_len: int = 0

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def page_count(self) -> int:
        return len(self._pages)

    @property
    def last_page_len(self) -> int:
        return self._last_page_len

    @property
    def total_tokens(self) -> int:
        if len(self._pages) == 0:
            return self._last_page_len
        return (len(self._pages) - 1) * self._page_size + self._last_page_len

    @property
    def page_ptrs(self) -> list[int]:
        return [p.ptr for p in self._pages if not p._released]

    def grow(self, num_tokens: int) -> None:
        self._adjust(num_tokens)

    def shrink(self, num_tokens: int) -> None:
        self._adjust(-num_tokens)

    def _adjust(self, num_tokens: int) -> None:
        if num_tokens == 0:
            return

        current_tokens = self.total_tokens
        new_total_tokens = current_tokens + num_tokens
        if new_total_tokens < 0:
            raise ValueError("Token count adjustment resulted in underflow")

        current_pages = len(self._pages)
        required_pages = math.ceil(new_total_tokens / self._page_size) if new_total_tokens > 0 else 0

        if required_pages > current_pages:
            new_pages_needed = required_pages - current_pages
            ptrs = allocate_resources(self._host_queue, KV_PAGE_TYPE, new_pages_needed)
            for ptr in ptrs:
                self._pages.append(KvPage(self._host_queue, ptr))
        elif required_pages < current_pages:
            pages_to_release = self._pages[required_pages:]
            self._pages = self._pages[:required_pages]
            for page in pages_to_release:
                page.release()

        last_page_len = new_total_tokens % self._page_size
        if last_page_len == 0 and new_total_tokens > 0:
            self._last_page_len = self._page_size
        else:
            self._last_page_len = last_page_len

    def import_pages_from_state(self, ptrs: list[int], last_page_len: int) -> None:
        self._pages = [KvPage(self._host_queue, ptr) for ptr in ptrs]
        self._last_page_len = last_page_len

    def release_all(self) -> None:
        for page in self._pages:
            page.release()
        self._pages.clear()
        self._last_page_len = 0

    def fork(self) -> tuple[KvPageManager, int]:
        """Fork with copy-on-write sharing. Only shares full pages."""
        kept_page_count = max(0, len(self._pages) - 1)
        kept_tokens = kept_page_count * self._page_size
        dropped_token_count = self.total_tokens - kept_tokens

        forked = KvPageManager(self._host_queue, self._page_size)
        forked._pages = self._pages[:kept_page_count]
        forked._last_page_len = self._page_size if kept_page_count > 0 else 0

        for page in forked._pages:
            page.ref()

        return forked, dropped_token_count

    def remove_page_at(self, index: int) -> None:
        removed = self._pages.pop(index)
        removed.release()

    def recalculate_last_page_len(self, total_tokens: int) -> None:
        last_page_len = total_tokens % self._page_size
        self._last_page_len = (
            self._page_size
            if last_page_len == 0 and total_tokens > 0
            else last_page_len
        )
