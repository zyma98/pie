"""
Hook-Based Operation Tracker

Uses PyTorch forward hooks to directly track tensor IDs flowing through operations.
This provides accurate operation → tensor mappings without relying on profiler memory data.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any

import torch
from torch import nn

# Import for type checking and runtime use
try:
    from .tracker import TensorLifecycleEvent
except ImportError:
    # Fallback for when running as script
    from tracker import TensorLifecycleEvent


class HookBasedTracker:
    """
    Tracks operations using forward hooks to capture exact tensor IDs.

    This is more reliable than profiler-based tracking because it directly
    observes which tensor objects flow through each operation.
    """

    def __init__(self, base_tracker):
        """Initialize with reference to base memory tracker."""
        self.base_tracker = base_tracker
        self.enabled = base_tracker.enabled

        # Storage for hook data
        self._hook_data = []
        self._hook_handles = []
        self._operation_log = []  # Operation-centric log with timestamps

    @contextmanager
    def track_model_with_hooks(self, model: nn.Module):
        """
        Track model forward pass using hooks.

        Args:
            model: PyTorch model to track

        Yields:
            None
        """
        if not self.enabled:
            yield
            return

        print("🪝 Hook-based tracker: Registering forward hooks on model")

        # Clear previous data
        self._hook_data.clear()
        self._hook_handles.clear()

        # Register hooks on all leaf modules
        hook_count = 0
        for name, module in model.named_modules():
            # Only hook leaf modules (those with no children)
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(self._create_hook(name))
                self._hook_handles.append(handle)
                hook_count += 1

        print(f"🪝 Registered {hook_count} forward hooks")

        # Capture a snapshot BEFORE forward pass to register existing tensors
        # (weights, kv_cache, etc.) so hooks can update their allocated_by field
        print("🪝 Capturing pre-forward snapshot to register existing tensors...")
        self.base_tracker._capture_snapshot("pre_forward_hook")

        try:
            yield
        finally:
            # Remove hooks
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()

            print(f"🪝 Captured {len(self._hook_data)} operation calls")

            # Capture post-forward snapshot to register NEW tensors created during forward
            print("🪝 Capturing post-forward snapshot to register new tensors...")
            self.base_tracker._capture_snapshot("post_forward_hook")

            # Process hook data
            self._process_hook_data()

    def _create_hook(self, module_name: str):
        """Create a forward hook function for a module."""
        def hook(module, input_tensors, output):
            # Capture timestamp when operation executes
            timestamp = datetime.now().isoformat()

            # Extract tensor IDs
            input_ids = []
            for inp in input_tensors:
                if isinstance(inp, torch.Tensor):
                    input_ids.append(id(inp))
                elif isinstance(inp, (tuple, list)):
                    for t in inp:
                        if isinstance(t, torch.Tensor):
                            input_ids.append(id(t))

            output_ids = []
            if isinstance(output, torch.Tensor):
                output_ids.append(id(output))
            elif isinstance(output, (tuple, list)):
                for t in output:
                    if isinstance(t, torch.Tensor):
                        output_ids.append(id(t))

            # Get shapes for validation
            input_shapes = []
            for inp in input_tensors:
                if isinstance(inp, torch.Tensor):
                    input_shapes.append(tuple(inp.shape))

            output_shape = None
            if isinstance(output, torch.Tensor):
                output_shape = tuple(output.shape)

            # Store hook data with timestamp
            self._hook_data.append({
                'timestamp': timestamp,
                'module_name': module_name,
                'module_type': module.__class__.__name__,
                'input_tensor_ids': input_ids,
                'output_tensor_ids': output_ids,
                'input_shapes': input_shapes,
                'output_shape': output_shape,
            })

        return hook

    def _process_hook_data(self):
        """
        Build operation-centric log from captured hook data.

        This creates a timeline of operation executions with input/output tensors.
        """
        print(f"\n📊 Processing {len(self._hook_data)} hook captures")

        # Build operation log (operation-centric view)
        self._operation_log = []
        for idx, hook_info in enumerate(self._hook_data):
            op_name = f"{hook_info['module_type']}.{hook_info['module_name']}"

            operation_entry = {
                'operation_id': idx + 1,
                'timestamp': hook_info['timestamp'],
                'name': op_name,
                'module_type': hook_info['module_type'],
                'module_name': hook_info['module_name'],
                'input_tensor_ids': hook_info['input_tensor_ids'],
                'output_tensor_ids': hook_info['output_tensor_ids'],
                'input_shapes': hook_info['input_shapes'],
                'output_shape': hook_info['output_shape'],
                'num_inputs': len(hook_info['input_tensor_ids']),
                'num_outputs': len(hook_info['output_tensor_ids']),
            }
            self._operation_log.append(operation_entry)

        # Store in base tracker for export
        if not hasattr(self.base_tracker, '_operation_log'):
            self.base_tracker._operation_log = []
        self.base_tracker._operation_log.extend(self._operation_log)

        # Also update tensor metadata (for backward compatibility)
        updated_allocations = 0
        for hook_info in self._hook_data:
            op_name = f"{hook_info['module_type']}.{hook_info['module_name']}"

            # Update output tensors - mark them as allocated by this operation
            for tensor_id in hook_info['output_tensor_ids']:
                if tensor_id in self.base_tracker._tensor_registry:
                    self.base_tracker._tensor_registry[tensor_id].allocated_by = op_name
                    updated_allocations += 1

        # Generate summary
        operations = set(f"{h['module_type']}.{h['module_name']}" for h in self._hook_data)
        print(f"✅ Built operation log with {len(self._operation_log)} entries")
        print(f"✅ Tracked {len(operations)} unique operations")
        print(f"✅ Updated {updated_allocations} tensor allocations")

        # Show some examples
        print("\n📋 Sample operation timeline:")
        for i, op in enumerate(self._operation_log[:5]):
            print(f"  {op['operation_id']}. [{op['timestamp']}] {op['name']}")
            print(f"     Inputs: {op['num_inputs']} tensor(s)")
            print(f"     Outputs: {op['num_outputs']} tensor(s)")

    def get_hook_data(self) -> list[dict[str, Any]]:
        """Get the captured hook data."""
        return self._hook_data


def create_hook_tracker(base_tracker):
    """Create a hook-based tracker from a base tracker."""
    return HookBasedTracker(base_tracker)
