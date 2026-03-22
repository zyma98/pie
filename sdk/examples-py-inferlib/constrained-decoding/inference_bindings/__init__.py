"""
inference_bindings - Thin Python wrapper over inferlib WIT-generated bindings.

Pure re-exports of WIT resource types and runtime functions.
Logically equivalent to Rust's inferlib-inference-bindings crate.
"""

from wit_world.imports import models as _models
from wit_world.imports import inference as _inference
from wit_world.imports import runtime as _runtime
from wit_world.imports import formatter as _formatter
from wit_world.imports import messaging as _messaging
from wit_world.imports import kvstore as _kvstore
from wit_world.imports import queues as _queues
from wit_world.imports.poll import Pollable, poll

# Models
Model = _models.Model
Tokenizer = _models.Tokenizer

# Inference
Context = _inference.Context
SamplerConfig = _inference.SamplerConfig
SamplerConfig_Greedy = _inference.SamplerConfig_Greedy
SamplerConfig_Multinomial = _inference.SamplerConfig_Multinomial
SamplerConfig_TopP = _inference.SamplerConfig_TopP
SamplerConfig_TopK = _inference.SamplerConfig_TopK
SamplerConfig_MinP = _inference.SamplerConfig_MinP
SamplerConfig_TopKTopP = _inference.SamplerConfig_TopKTopP
StopConfig = _inference.StopConfig
GenerateFuture = _inference.GenerateFuture
DecodeStepFuture = _inference.DecodeStepFuture
FlushFuture = _inference.FlushFuture

# Queues / ForwardPass
Queue = _queues.Queue
ForwardPass = _queues.ForwardPass
ForwardPassResult = _queues.ForwardPassResult
Distribution = _queues.Distribution

# Formatter
ChatFormatter = _formatter.ChatFormatter
ToolCall = _formatter.ToolCall

# Runtime
get_arguments = _runtime.get_arguments
set_return = _runtime.set_return
get_version = _runtime.get_version
get_instance_id = _runtime.get_instance_id

# Messaging
send = _messaging.send
receive = _messaging.receive
send_blob = _messaging.send_blob
receive_blob = _messaging.receive_blob
broadcast = _messaging.broadcast
subscribe = _messaging.subscribe

# KV store
store_get = _kvstore.store_get
store_set = _kvstore.store_set
store_delete = _kvstore.store_delete
store_exists = _kvstore.store_exists
store_list_keys = _kvstore.store_list_keys
