"""
Memory profiling tracker for PyTorch tensors with operation context tracking.

This module provides automatic memory tracking for all allocated tensors
(CPU, MPS, CUDA) and exports snapshots to JSON files for analysis.
It also tracks which operations allocate which tensors using context managers.
"""

from __future__ import annotations

import gc
import json
import threading
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


@dataclass
class TensorInfo:
    """Information about a single tensor allocation."""

    tensor_id: int  # Unique ID for this tensor (id(tensor))
    size_bytes: int
    size_mb: float
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    allocated_by: str | None = None  # Which operation allocated this tensor
    purpose: str | None = None  # buffer, output, weight, cache, etc.
    is_persistent: bool = False  # Does it live across operations?
    is_reusable: bool = False  # Can memory be reused?


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a specific point in time."""

    timestamp: str
    checkpoint_name: str
    total_bytes: int
    total_mb: float
    device_breakdown: dict[str, dict[str, Any]]
    top_allocations: list[dict[str, Any]]
    tensor_count: int


@dataclass
class TensorLifecycleEvent:
    """Records an event in a tensor's lifecycle."""

    tensor_id: int
    event_type: str  # 'allocated', 'accessed_read', 'accessed_write', 'deallocated'
    operation: str
    timestamp: str
    size_mb: float
    shape: tuple[int, ...]


@dataclass
class OperationNode:
    """Represents an operation in the execution graph."""

    name: str
    invocations: int = 0
    total_memory_delta_mb: float = 0.0
    tensors_allocated: list[int] = None  # Tensor IDs
    tensors_read: list[int] = None  # Tensor IDs
    tensors_written: list[int] = None  # Tensor IDs
    child_operations: list[str] = None  # For nested ops

    def __post_init__(self):
        if self.tensors_allocated is None:
            self.tensors_allocated = []
        if self.tensors_read is None:
            self.tensors_read = []
        if self.tensors_written is None:
            self.tensors_written = []
        if self.child_operations is None:
            self.child_operations = []


class MemoryTracker:
    """
    Tracks PyTorch tensor memory allocations across all devices.

    This tracker periodically scans all live PyTorch tensors and records
    memory usage statistics. It can export snapshots to JSON files.
    """

    def __init__(
        self, output_dir: str = ".", enabled: bool = False, interval: float = 5.0
    ):
        """
        Initialize the memory tracker.

        Args:
            output_dir: Directory where memory snapshots will be saved
            enabled: Whether memory tracking is enabled
            interval: Interval in seconds between automatic snapshots (default: 5.0)
        """
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: list[MemorySnapshot] = []
        self._lock = threading.Lock()
        self._interval = interval
        self._stop_event = threading.Event()
        self._tracker_thread: threading.Thread | None = None

        # Operation context tracking (thread-local)
        self._operation_context = threading.local()
        self._operation_stack: list[str] = []  # For nested operations
        self._operation_allocations: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Tensor lifecycle and operation graph tracking
        self._tensor_registry: dict[int, TensorInfo] = {}  # tensor_id -> TensorInfo
        self._tensor_lifecycle: dict[int, list[TensorLifecycleEvent]] = defaultdict(
            list
        )
        self._operation_graph: dict[str, OperationNode] = (
            {}
        )  # operation_name -> OperationNode
        self._tensor_last_seen: dict[int, str] = {}  # Track when tensors were last seen

        # PyTorch operation tracking
        self._pytorch_ops: list[dict[str, Any]] = []  # List of all PyTorch ops executed
        self._op_hook_handle = None

        if self.enabled:
            print("✅ Memory tracking enabled")
            print(f"   Snapshots will be saved to: {self.output_dir.absolute()}")
            print(f"   Snapshot interval: {interval}s")
            print("   Operation tracking: ON (use track_operation() context manager)")
            self._start_periodic_tracking()

    @contextmanager
    def track_pytorch_ops(self):
        """
        Context manager to track low-level PyTorch operations (gemm, matmul, add, etc.).

        This uses PyTorch's profiler to capture every operation and build a complete
        operation graph showing: op_name → input_tensors → output_tensors

        Usage:
            with tracker.track_pytorch_ops():
                output = model.forward(input)
        """
        if not self.enabled:
            yield
            return

        # Clear previous ops
        self._pytorch_ops.clear()

        # Capture memory snapshot before
        with self._lock:
            pre_snapshot = self._capture_snapshot("pytorch_ops_pre")

        # Use PyTorch profiler with memory tracking
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # Stack traces are expensive
            with_modules=False,
        ) as prof:
            yield prof

        # Capture memory snapshot after
        with self._lock:
            post_snapshot = self._capture_snapshot("pytorch_ops_post")

        # Extract operation information from profiler
        self._extract_pytorch_ops_from_profiler(prof)

    def _extract_pytorch_ops_from_profiler(self, prof):
        """Extract operation information from PyTorch profiler."""
        try:
            for evt in prof.key_averages():
                # Skip profiler overhead events
                if "profiler" in evt.key.lower():
                    continue

                op_info = {
                    "op_name": evt.key,
                    "count": evt.count,
                    "cpu_time_us": evt.cpu_time_total,
                    "cuda_time_us": (
                        evt.cuda_time_total if hasattr(evt, "cuda_time_total") else 0
                    ),
                    "cpu_memory_mb": (
                        evt.cpu_memory_usage / (1024 * 1024)
                        if hasattr(evt, "cpu_memory_usage")
                        else 0
                    ),
                    "cuda_memory_mb": (
                        evt.cuda_memory_usage / (1024 * 1024)
                        if hasattr(evt, "cuda_memory_usage")
                        else 0
                    ),
                    "input_shapes": (
                        evt.input_shapes if hasattr(evt, "input_shapes") else []
                    ),
                }
                self._pytorch_ops.append(op_info)
        except Exception as e:
            print(f"⚠️  Failed to extract PyTorch ops: {e}")

    @contextmanager
    def track_operation(self, operation_name: str):
        """
        Context manager to track memory allocations and tensor access within an operation.

        This builds an operation graph showing:
        - Which tensors were allocated
        - Which tensors were read/written
        - Memory deltas
        - Parent-child operation relationships

        Usage:
            with tracker.track_operation("model_forward"):
                output = model.forward(input)

        Args:
            operation_name: Name of the operation being tracked
        """
        if not self.enabled:
            yield
            return

        # Initialize operation node if first time seeing this operation
        if operation_name not in self._operation_graph:
            self._operation_graph[operation_name] = OperationNode(name=operation_name)

        # Track parent-child relationship
        if len(self._operation_stack) > 0:
            parent_op = self._operation_stack[-1]
            if parent_op in self._operation_graph:
                if (
                    operation_name
                    not in self._operation_graph[parent_op].child_operations
                ):
                    self._operation_graph[parent_op].child_operations.append(
                        operation_name
                    )

        # Capture memory before operation
        with self._lock:
            pre_snapshot = self._capture_snapshot(f"{operation_name}_pre")
            pre_tensor_ids = set(self._tensor_registry.keys())

        # Set operation context
        self._operation_stack.append(operation_name)

        try:
            yield
        finally:
            # Clear operation context
            if self._operation_stack and self._operation_stack[-1] == operation_name:
                self._operation_stack.pop()

            # Capture memory after operation
            with self._lock:
                post_snapshot = self._capture_snapshot(f"{operation_name}_post")
                post_tensor_ids = set(self._tensor_registry.keys())

                # Analyze tensor changes
                new_tensor_ids = post_tensor_ids - pre_tensor_ids
                deallocated_tensor_ids = pre_tensor_ids - post_tensor_ids
                existing_tensor_ids = pre_tensor_ids & post_tensor_ids

                # Update operation node
                op_node = self._operation_graph[operation_name]
                op_node.invocations += 1

                # Record new allocations
                for tensor_id in new_tensor_ids:
                    if tensor_id in self._tensor_registry:
                        op_node.tensors_allocated.append(tensor_id)
                        self._record_lifecycle_event(
                            tensor_id,
                            "allocated",
                            operation_name,
                            self._tensor_registry[tensor_id].size_mb,
                            self._tensor_registry[tensor_id].shape,
                        )

                # Record tensor accesses (existing tensors = read)
                for tensor_id in existing_tensor_ids:
                    if tensor_id not in op_node.tensors_read:
                        op_node.tensors_read.append(tensor_id)
                    self._record_lifecycle_event(
                        tensor_id,
                        "accessed_read",
                        operation_name,
                        self._tensor_registry[tensor_id].size_mb,
                        self._tensor_registry[tensor_id].shape,
                    )

                # Record deallocations
                for tensor_id in deallocated_tensor_ids:
                    if tensor_id in self._tensor_registry:
                        self._record_lifecycle_event(
                            tensor_id,
                            "deallocated",
                            operation_name,
                            self._tensor_registry[tensor_id].size_mb,
                            self._tensor_registry[tensor_id].shape,
                        )

                # Calculate memory delta
                memory_delta_mb = post_snapshot.total_mb - pre_snapshot.total_mb
                tensor_delta = post_snapshot.tensor_count - pre_snapshot.tensor_count
                op_node.total_memory_delta_mb += memory_delta_mb

                # Record operation allocation info
                self._operation_allocations[operation_name].append(
                    {
                        "memory_delta_mb": memory_delta_mb,
                        "tensor_delta": tensor_delta,
                        "tensors_allocated": len(new_tensor_ids),
                        "tensors_deallocated": len(deallocated_tensor_ids),
                        "tensors_accessed": len(existing_tensor_ids),
                        "pre_total_mb": pre_snapshot.total_mb,
                        "post_total_mb": post_snapshot.total_mb,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

    def _record_lifecycle_event(
        self,
        tensor_id: int,
        event_type: str,
        operation: str,
        size_mb: float,
        shape: tuple,
    ):
        """Record a lifecycle event for a tensor."""
        event = TensorLifecycleEvent(
            tensor_id=tensor_id,
            event_type=event_type,
            operation=operation,
            timestamp=datetime.now().isoformat(),
            size_mb=size_mb,
            shape=shape,
        )
        self._tensor_lifecycle[tensor_id].append(event)

    def checkpoint(self, name: str) -> None:
        """
        Record a memory checkpoint with the given name.

        Args:
            name: Descriptive name for this checkpoint (e.g., "after_model_load")
        """
        if not self.enabled:
            return

        with self._lock:
            snapshot = self._capture_snapshot(name)
            self.snapshots.append(snapshot)

            # Print summary to console
            print(f"\n📊 Memory Checkpoint: {name}")
            print(
                f"   Total: {snapshot.total_mb:.2f} MB "
                f"across {snapshot.tensor_count} tensors"
            )
            for device, stats in snapshot.device_breakdown.items():
                print(
                    f"   {device}: {stats['total_mb']:.2f} MB "
                    f"({stats['count']} tensors)"
                )

    def _capture_snapshot(self, checkpoint_name: str) -> MemorySnapshot:
        """Capture current memory state across all devices."""
        # Get current operation context
        current_operation = self._operation_stack[-1] if self._operation_stack else None

        # Collect all tensors
        tensors: list[TensorInfo] = []
        device_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_bytes": 0, "total_mb": 0.0, "count": 0, "tensors": []}
        )

        current_tensor_ids = set()
        previous_tensor_ids = set(self._tensor_registry.keys())

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    tensor_id = id(obj)
                    current_tensor_ids.add(tensor_id)

                    size_bytes = obj.element_size() * obj.nelement()
                    size_mb = size_bytes / (1024**2)
                    device = str(obj.device)
                    shape = tuple(obj.shape)

                    # Classify tensor purpose
                    purpose = self._classify_tensor_purpose(obj, shape, device)
                    is_persistent = self._is_persistent_tensor(tensor_id, size_mb)
                    is_reusable = self._is_reusable_tensor(obj, purpose)

                    # Check if this is a new tensor or update to registry
                    if tensor_id not in self._tensor_registry:
                        # New tensor - register it and record allocation
                        tensor_info = TensorInfo(
                            tensor_id=tensor_id,
                            size_bytes=size_bytes,
                            size_mb=size_mb,
                            shape=shape,
                            dtype=str(obj.dtype),
                            device=device,
                            requires_grad=obj.requires_grad,
                            allocated_by=current_operation or "unknown",
                            purpose=purpose,
                            is_persistent=is_persistent,
                            is_reusable=is_reusable,
                        )
                        self._tensor_registry[tensor_id] = tensor_info

                        # Record allocation event
                        self._record_lifecycle_event(
                            tensor_id,
                            "allocated",
                            current_operation or "background",
                            size_mb,
                            shape,
                        )
                    else:
                        # Existing tensor - update info
                        tensor_info = self._tensor_registry[tensor_id]

                        # Record access if we haven't seen it in this snapshot yet
                        if self._tensor_last_seen.get(tensor_id) != checkpoint_name:
                            self._record_lifecycle_event(
                                tensor_id,
                                "accessed_read",
                                current_operation or "background",
                                size_mb,
                                shape,
                            )

                    tensors.append(tensor_info)

                    # Update device statistics
                    device_stats[device]["total_bytes"] += size_bytes
                    device_stats[device]["total_mb"] += size_mb
                    device_stats[device]["count"] += 1
                    device_stats[device]["tensors"].append(asdict(tensor_info))

                    # Track last seen
                    self._tensor_last_seen[tensor_id] = checkpoint_name

            except Exception:
                # Skip objects that can't be inspected
                pass

        # Detect deallocated tensors
        deallocated_tensor_ids = previous_tensor_ids - current_tensor_ids
        for tensor_id in deallocated_tensor_ids:
            if tensor_id in self._tensor_registry:
                tensor_info = self._tensor_registry[tensor_id]
                self._record_lifecycle_event(
                    tensor_id,
                    "deallocated",
                    current_operation or "background",
                    tensor_info.size_mb,
                    tensor_info.shape,
                )

        # Calculate totals
        total_bytes = sum(t.size_bytes for t in tensors)
        total_mb = total_bytes / (1024**2)

        # Get top 10 largest allocations
        top_tensors = sorted(tensors, key=lambda t: t.size_bytes, reverse=True)[:10]
        top_allocations = [asdict(t) for t in top_tensors]

        # Convert device_stats to regular dict
        device_breakdown = dict(device_stats)

        return MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            checkpoint_name=checkpoint_name,
            total_bytes=total_bytes,
            total_mb=total_mb,
            device_breakdown=device_breakdown,
            top_allocations=top_allocations,
            tensor_count=len(tensors),
        )

    def save_snapshot(self, filename: str | None = None) -> str:
        """
        Save all recorded snapshots to a JSON file.

        Args:
            filename: Optional custom filename. If None, generates timestamped filename.

        Returns:
            Path to the saved file
        """
        if not self.enabled:
            return ""

        with self._lock:
            if not self.snapshots:
                print("⚠️  No memory snapshots to save")
                return ""

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"memory_profile_{timestamp}.json"

            output_path = self.output_dir / filename

            # Prepare data for export
            export_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "num_snapshots": len(self.snapshots),
                    "operation_tracking_enabled": bool(self._operation_allocations),
                    "pytorch_ops_tracking_enabled": bool(self._pytorch_ops),
                    "tensor_lifecycle_tracking_enabled": bool(self._tensor_lifecycle),
                },
                "snapshots": [asdict(snapshot) for snapshot in self.snapshots],
            }

            # Add operation allocations if any were tracked
            if self._operation_allocations:
                export_data["operation_allocations"] = dict(self._operation_allocations)
                export_data["operation_summary"] = self._generate_operation_summary()

            # Add operation graph if tracked
            if self._operation_graph:
                export_data["operation_graph"] = self._export_operation_graph()

            # Add tensor lifecycle if tracked
            if self._tensor_lifecycle:
                export_data["tensor_lifecycle"] = self._export_tensor_lifecycle()

            # Add PyTorch operations if tracked
            if self._pytorch_ops:
                export_data["pytorch_operations"] = self._pytorch_ops

            # Add operation log if tracked (from hook-based tracker)
            if hasattr(self, '_operation_log') and self._operation_log:
                export_data["operation_log"] = self._operation_log
                export_data["metadata"]["operation_log_enabled"] = True

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)

            print(f"\n💾 Memory profile saved to: {output_path.absolute()}")
            if self._operation_allocations:
                print(f"   • {len(self._operation_allocations)} operations tracked")
            if self._operation_graph:
                print(f"   • Operation graph with {len(self._operation_graph)} nodes")
            if self._tensor_lifecycle:
                print(
                    f"   • Lifecycle tracked for {len(self._tensor_lifecycle)} tensors"
                )
            if self._pytorch_ops:
                print(f"   • {len(self._pytorch_ops)} PyTorch operations captured")
            if hasattr(self, '_operation_log') and self._operation_log:
                print(f"   • Operation log with {len(self._operation_log)} entries")
            return str(output_path.absolute())

    def _export_operation_graph(self) -> dict[str, Any]:
        """Export operation graph as a dictionary."""
        graph = {}
        for op_name, node in self._operation_graph.items():
            graph[op_name] = {
                "invocations": node.invocations,
                "total_memory_delta_mb": round(node.total_memory_delta_mb, 2),
                "tensors_allocated": len(set(node.tensors_allocated)),
                "tensors_read": len(set(node.tensors_read)),
                "tensors_written": len(set(node.tensors_written)),
                "child_operations": node.child_operations,
                "tensor_details": {
                    "allocated": [
                        self._tensor_registry[tid].purpose
                        for tid in set(node.tensors_allocated)
                        if tid in self._tensor_registry
                    ],
                    "read": [
                        self._tensor_registry[tid].purpose
                        for tid in set(node.tensors_read)
                        if tid in self._tensor_registry
                    ],
                },
            }
        return graph

    def _export_tensor_lifecycle(self) -> dict[str, Any]:
        """Export tensor lifecycle events."""
        lifecycle = {}
        for tensor_id, events in self._tensor_lifecycle.items():
            if tensor_id in self._tensor_registry:
                tensor_info = self._tensor_registry[tensor_id]
                lifecycle[str(tensor_id)] = {
                    "purpose": tensor_info.purpose,
                    "size_mb": tensor_info.size_mb,
                    "shape": list(tensor_info.shape),
                    "is_persistent": tensor_info.is_persistent,
                    "is_reusable": tensor_info.is_reusable,
                    "events": [asdict(event) for event in events],
                }
        return lifecycle

    def _generate_operation_summary(self) -> dict[str, Any]:
        """Generate a summary of operation-level memory allocations."""
        summary = {}
        for op_name, allocations in self._operation_allocations.items():
            total_delta = sum(a["memory_delta_mb"] for a in allocations)
            avg_delta = total_delta / len(allocations) if allocations else 0
            max_delta = max((a["memory_delta_mb"] for a in allocations), default=0)

            summary[op_name] = {
                "invocation_count": len(allocations),
                "total_memory_delta_mb": round(total_delta, 2),
                "avg_memory_delta_mb": round(avg_delta, 2),
                "max_memory_delta_mb": round(max_delta, 2),
            }
        return summary

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of all recorded snapshots.

        Returns:
            Dictionary containing summary statistics
        """
        if not self.enabled or not self.snapshots:
            return {}

        with self._lock:
            summary = {
                "num_snapshots": len(self.snapshots),
                "checkpoints": [s.checkpoint_name for s in self.snapshots],
                "peak_memory_mb": max(s.total_mb for s in self.snapshots),
                "peak_memory_checkpoint": max(
                    self.snapshots, key=lambda s: s.total_mb
                ).checkpoint_name,
                "final_memory_mb": (
                    self.snapshots[-1].total_mb if self.snapshots else 0
                ),
            }

            # Add operation tracking summary if available
            if self._operation_allocations:
                summary["operations"] = self._generate_operation_summary()

            return summary

    def reset(self) -> None:
        """Clear all recorded snapshots."""
        with self._lock:
            self.snapshots.clear()

    def _start_periodic_tracking(self) -> None:
        """Start background thread for periodic memory snapshots."""
        self._tracker_thread = threading.Thread(
            target=self._periodic_snapshot_loop, daemon=True, name="MemoryTracker"
        )
        self._tracker_thread.start()

    def _periodic_snapshot_loop(self) -> None:
        """Background loop that captures memory snapshots periodically."""
        snapshot_count = 0
        while not self._stop_event.is_set():
            # Wait for the interval, or until stop is signaled
            if self._stop_event.wait(timeout=self._interval):
                break

            # Capture snapshot
            snapshot_count += 1
            snapshot_name = f"snapshot_{snapshot_count:04d}"
            with self._lock:
                snapshot = self._capture_snapshot(snapshot_name)
                self.snapshots.append(snapshot)

    def _classify_tensor_purpose(
        self, tensor: torch.Tensor, shape: tuple, device: str
    ) -> str:
        """Classify what this tensor is used for."""
        # KV cache detection: large 5D tensors with specific shape pattern
        if (
            len(shape) == 5 and shape[0] > 4096
        ):  # [max_cache_len, batch, num_kv_heads, num_heads_per_group, head_dim]
            return "kv_cache"

        # Weight detection: requires_grad or very large
        if tensor.requires_grad:
            if tensor.numel() > 1000000:  # > 1M parameters
                return "weight"
            return "gradient"

        # Embedding detection: 2D with large vocabulary size
        if len(shape) == 2 and shape[0] > 10000:
            return "embedding"

        # Intermediate buffer: modest size, temporary
        if len(shape) >= 2:
            return "buffer"

        return "unknown"

    def _is_persistent_tensor(self, tensor_id: int, size_mb: float) -> bool:
        """Determine if tensor persists across operations."""
        # If we've seen this tensor before, it's persistent
        if tensor_id in self._tensor_last_seen:
            return True

        # Large tensors (>100MB) are likely persistent (weights, cache)
        if size_mb > 100:
            return True

        return False

    def _is_reusable_tensor(self, tensor: torch.Tensor, purpose: str) -> bool:
        """Determine if tensor memory can be reused."""
        # Weights and cache are not reusable
        if purpose in ["weight", "kv_cache", "embedding"]:
            return False

        # Gradients are reusable (overwritten each backward pass)
        if purpose == "gradient":
            return True

        # Buffers without grad are reusable
        if purpose == "buffer" and not tensor.requires_grad:
            return True

        return False

    def stop(self) -> None:
        """Stop periodic tracking and save final snapshot."""
        if not self.enabled:
            return

        self._stop_event.set()
        if self._tracker_thread and self._tracker_thread.is_alive():
            self._tracker_thread.join(timeout=2.0)

        # Save final snapshot
        self.save_snapshot()


# Global singleton instance
_TRACKER: MemoryTracker | None = None


def initialize_memory_tracker(output_dir: str = ".", enabled: bool = False) -> None:
    """
    Initialize the global memory tracker.

    Args:
        output_dir: Directory where memory snapshots will be saved
        enabled: Whether memory tracking is enabled
    """
    global _TRACKER  # pylint: disable=global-statement
    _TRACKER = MemoryTracker(output_dir=output_dir, enabled=enabled)


def get_memory_tracker() -> MemoryTracker:
    """Get the global memory tracker instance."""
    global _TRACKER  # pylint: disable=global-statement
    if _TRACKER is None:
        # Initialize with disabled tracker if not initialized
        _TRACKER = MemoryTracker(enabled=False)
    return _TRACKER


def memory_checkpoint(name: str) -> nullcontext[None]:
    """
    Context manager or direct call to record a memory checkpoint.

    Can be used as:
        memory_checkpoint("checkpoint_name")
    or:
        with memory_checkpoint("checkpoint_name"):
            # code here

    Args:
        name: Descriptive name for this checkpoint
    """
    tracker = get_memory_tracker()
    tracker.checkpoint(name)
    return nullcontext()


def save_memory_profile(filename: str | None = None) -> str:
    """
    Save recorded memory snapshots to JSON file.

    Args:
        filename: Optional custom filename

    Returns:
        Path to saved file
    """
    tracker = get_memory_tracker()
    return tracker.save_snapshot(filename)


def get_memory_summary() -> dict[str, Any]:
    """Get summary of recorded memory snapshots."""
    tracker = get_memory_tracker()
    return tracker.get_summary()


def stop_memory_tracker() -> None:
    """Stop periodic tracking and save final snapshot."""
    tracker = get_memory_tracker()
    tracker.stop()
