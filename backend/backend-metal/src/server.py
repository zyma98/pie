"""Metal backend server entrypoint reusing shared infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path

import fire

BACKEND_PYTHON_PATH = Path(__file__).resolve().parents[2] / "backend-python"
if str(BACKEND_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PYTHON_PATH))

from model_loader import load_model as load_model_common
from server_common import build_config, print_config, start_service

from handler import Handler
from model_factory import create_model_and_fusion_map


def main(
    model: str,
    host: str = "localhost",
    port: int = 10123,
    controller_host: str = "localhost",
    controller_port: int = 9123,
    auth_token: str | None = None,
    cache_dir: str | None = None,
    kv_page_size: int = 16,
    max_dist_size: int = 64,
    max_num_kv_pages: int = 1024,
    max_num_embeds: int = 128,
    max_num_adapters: int = 48,
    max_adapter_rank: int = 8,
    device: str = "mps",
    dtype: str = "bfloat16",
):
    config = build_config(
        model=model,
        host=host,
        port=port,
        controller_host=controller_host,
        controller_port=controller_port,
        auth_token=auth_token,
        cache_dir=cache_dir,
        kv_page_size=kv_page_size,
        max_dist_size=max_dist_size,
        max_num_kv_pages=max_num_kv_pages,
        max_num_embeds=max_num_embeds,
        max_num_adapters=max_num_adapters,
        max_adapter_rank=max_adapter_rank,
        device=device,
        dtype=dtype,
    )

    print_config(config)

    model_instance, model_metadata = load_model_common(
        config,
        create_model_and_fusion_map,
    )

    start_service(
        config=config,
        handler_cls=Handler,
        model=model_instance,
        model_info=model_metadata,
    )


if __name__ == "__main__":
    fire.Fire(main)
