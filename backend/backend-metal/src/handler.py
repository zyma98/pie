"""Metal backend Handler implementation."""

from __future__ import annotations

import sys
from pathlib import Path
import torch

# Ensure the Python backend modules are importable so we can reuse Handler logic
BACKEND_PYTHON_PATH = Path(__file__).resolve().parents[2] / "backend-python"
if str(BACKEND_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PYTHON_PATH))

from handler_common import Handler as BaseHandler
from config.common import ModelInfo

# Import Metal-specific operations that plug into the shared Handler
from metal_ops import MetalOps


class MetalHandler(BaseHandler):
    """Metal backend handler that wires MetalOps into the shared Handler base."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_info: ModelInfo,
        kv_page_size: int,
        max_dist_size: int,
        max_num_kv_pages: int,
        max_num_embeds: int,
        max_num_adapters: int,
        max_adapter_rank: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        metal_ops = MetalOps()
        super().__init__(
            model=model,
            model_info=model_info,
            ops=metal_ops,
            kv_page_size=kv_page_size,
            max_dist_size=max_dist_size,
            max_num_kv_pages=max_num_kv_pages,
            max_num_embeds=max_num_embeds,
            max_num_adapters=max_num_adapters,
            max_adapter_rank=max_adapter_rank,
            dtype=dtype,
            device=device,
        )

        print("✅ MetalHandler initialized with Metal backend")
        print(f"   Metal kernels available: {metal_ops.available}")


# Alias that mirrors backend-python/handler.py so imports can use `from handler import Handler`
Handler = MetalHandler

__all__ = ["Handler", "MetalHandler"]
