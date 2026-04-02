"""Messaging interface implementation -- passthrough to host APIs."""

from wit_world.imports import message as _message
from wit_world.imports import inferlet_core_common as _common


class Messaging:
    def send(self, message: str) -> None:
        _message.send(message)

    def receive(self) -> str:
        result = _message.receive()
        while True:
            pollable = result.pollable()
            pollable.block()
            value = result.get()
            if value is not None:
                return value

    def send_blob(self, data: bytes) -> None:
        blob = _common.Blob(data)
        _message.send_blob(blob)

    def receive_blob(self) -> bytes:
        result = _message.receive_blob()
        while True:
            pollable = result.pollable()
            pollable.block()
            blob = result.get()
            if blob is not None:
                return bytes(blob.read(0, blob.size()))

    def broadcast(self, topic: str, message: str) -> None:
        _message.broadcast(topic, message)

    def subscribe(self, topic: str) -> str:
        subscription = _message.subscribe(topic)
        while True:
            pollable = subscription.pollable()
            pollable.block()
            value = subscription.get()
            if value is not None:
                return value
