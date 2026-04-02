# =============================================================================
# Entry point for the inference-py library component.
#
# componentize-py has specific conventions for how it discovers Python
# implementations of exported WIT interfaces. This file is the "app module"
# passed to componentize-py, and its job is to make all implementations
# discoverable. The rules are described below.
#
# =============================================================================
# RULE 1: File naming -- each .py file must match the WIT interface name.
#
#   For each exported WIT interface, componentize-py looks for a Python
#   module whose name is the snake_case form of the WIT interface name.
#   For example:
#
#     WIT interface "models"     -> models.py
#     WIT interface "kvstore"    -> kvstore.py
#     WIT interface "formatter"  -> formatter.py
#
# =============================================================================
# RULE 2: Two kinds of exported interfaces require different Python patterns.
#
#   componentize-py generates a Protocol class for each exported interface
#   in wit_world/exports/__init__.py. How you implement depends on whether
#   the interface contains resources or only freestanding functions:
#
#   (a) Resource-only interfaces (or interfaces with resources + functions):
#       The Protocol class in __init__.py is empty (`pass`). componentize-py
#       discovers resource implementations by looking for classes in the
#       matching module that subclass the Protocol from
#       wit_world/exports/<interface>.py.
#
#       Example: WIT "models" has resources `model` and `tokenizer`.
#       componentize-py generates:
#         - wit_world/exports/__init__.py:  class Models(Protocol): pass
#         - wit_world/exports/models.py:    class Model(Protocol): ...
#                                           class Tokenizer(Protocol): ...
#       You implement by subclassing in models.py:
#         from wit_world.exports.models import Model as ModelBase
#         class Model(ModelBase): ...
#
#       For these, a plain `import models` in app.py suffices -- componentize-py
#       scans the module for classes matching the resource Protocol names.
#
#   (b) Freestanding-function-only interfaces (no resources):
#       The Protocol class in __init__.py has instance methods corresponding
#       to each WIT function (with `self` as the first parameter).
#       componentize-py looks for a class ON THE APP MODULE (i.e. as a
#       top-level attribute of app.py) whose name matches the PascalCase
#       protocol name.
#
#       Example: WIT "runtime" has functions `get-version`, `set-return`, etc.
#       componentize-py generates:
#         - wit_world/exports/__init__.py:  class Runtime(Protocol):
#                                               def get_version(self) -> str: ...
#                                               def set_return(self, value: str): ...
#       You implement by defining a class with the EXACT name "Runtime":
#         class Runtime:
#             def get_version(self) -> str: ...
#             def set_return(self, value: str) -> None: ...
#
#       CRITICAL: this class must be importable as `app.Runtime`, so in app.py
#       you must use `from runtime import Runtime` (NOT `import runtime`).
#       A plain `import runtime` would make it accessible as `app.runtime.Runtime`
#       but componentize-py specifically looks for `app.Runtime`.
#
#   Summary of which pattern applies to each interface in this component:
#
#     Interface    | Has Resources?                  | Protocol class  | Pattern
#     -------------|---------------------------------|-----------------|-----------------------------
#     models       | Yes (Model, Tokenizer)          | Models: pass    | import models
#     queues       | Yes (Queue, ForwardPass)        | Queues: pass    | import queues
#     inference    | Yes (Context, futures)          | Inference: pass | import inference
#     formatter    | Yes (ChatFormatter)             | Formatter: pass | import formatter
#     runtime      | No (freestanding functions)     | Runtime: ...    | from runtime import Runtime
#     messaging    | No (freestanding functions)     | Messaging: ...  | from messaging import Messaging
#     kvstore      | No (freestanding functions)     | Kvstore: ...    | from kvstore import Kvstore
#
# =============================================================================
# RULE 3: Class naming for freestanding-function interfaces.
#
#   The class name must be the PascalCase version of the WIT interface name:
#
#     WIT interface "runtime"    -> class Runtime
#     WIT interface "messaging"  -> class Messaging
#     WIT interface "kvstore"    -> class Kvstore
#
#   Each WIT function becomes a method with `self`, using snake_case:
#     WIT: get-version: func() -> string     -> def get_version(self) -> str
#     WIT: set-return: func(value: string)   -> def set_return(self, value: str)
#
# =============================================================================
# RULE 4: Class naming for resources.
#
#   Resource class names are the PascalCase version of the WIT resource name,
#   and must subclass the Protocol from wit_world/exports/<interface>.py:
#
#     WIT resource "model"          -> class Model(ModelBase)
#     WIT resource "chat-formatter" -> class ChatFormatter(ChatFormatterBase)
#     WIT resource "forward-pass"   -> class ForwardPass(ForwardPassBase)
#
#   Static WIT functions become @classmethod, instance functions become methods.
#   WIT constructors become __init__.
#
# =============================================================================
# RULE 5: Import module naming for host API imports.
#
#   componentize-py generates import modules under wit_world/imports/ using
#   the SHORT interface name by default, unless that would cause a clash
#   with another imported interface. In that case, it qualifies with the
#   full package path (namespace_package_interface).
#
#   In this component, two packages both export a "common" interface:
#     inferlet:core/common    -> inferlet_core_common  (qualified)
#     inferlet:adapter/common -> inferlet_adapter_common (qualified)
#
#   All other imports use just the short name:
#     inferlet:core/runtime   -> runtime
#     inferlet:core/tokenize  -> tokenize
#     inferlet:core/forward   -> forward
#     inferlet:core/kvs       -> kvs
#     inferlet:core/message   -> message
#     inferlet:zo/evolve      -> evolve
#     inferlet:image/image    -> image
#     wasi:io/poll@0.2.0      -> poll
#
# =============================================================================

import models
import queues
import inference
import formatter

from runtime import Runtime
from messaging import Messaging
from kvstore import Kvstore
