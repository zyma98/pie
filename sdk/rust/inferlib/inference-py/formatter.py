"""Formatter interface implementation -- ChatFormatter resource."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wit_world.exports.formatter import (
    ChatFormatter as ChatFormatterBase,
    ToolCall as WitToolCall,
)

try:
    from jinja2 import Environment, BaseLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


@dataclass
class Message:
    role: str
    content: str
    reasoning_content: str | None = None
    tool_calls: list[WitToolCall] | None = None


class ChatFormatter(ChatFormatterBase):
    def __init__(self, template: str):
        self._messages: list[Message] = []
        self._template_str = template
        self._template: Any = None

        if HAS_JINJA2 and self._template_str:
            env = Environment(loader=BaseLoader())
            env.globals["raise_exception"] = self._raise_exception
            try:
                self._template = env.from_string(self._template_str)
            except Exception:
                self._template = None

    @staticmethod
    def _raise_exception(msg: str) -> None:
        raise ValueError(msg)

    def add_system(self, content: str) -> None:
        self._messages.append(Message(role="system", content=content))

    def add_user(self, content: str) -> None:
        self._messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self.add_assistant_response(content, None, None)

    def add_assistant_response(
        self,
        content: str,
        reasoning: str | None,
        tool_calls: list[WitToolCall] | None,
    ) -> None:
        self._messages.append(
            Message(
                role="assistant",
                content=content,
                reasoning_content=reasoning,
                tool_calls=tool_calls,
            )
        )

    def add_tool(self, content: str) -> None:
        self._messages.append(Message(role="tool", content=content))

    def has_messages(self) -> bool:
        return len(self._messages) > 0

    def clear(self) -> None:
        self._messages.clear()

    def render(self, add_generation_prompt: bool, begin_of_sequence: bool) -> str:
        messages_dict = self._messages_to_dict()

        if self._template is not None:
            try:
                return self._template.render(
                    messages=messages_dict,
                    add_generation_prompt=add_generation_prompt,
                    begin_of_sequence=begin_of_sequence,
                    bos_token="",
                    eos_token="",
                )
            except Exception:
                pass

        return self._format_fallback(messages_dict, add_generation_prompt)

    def _messages_to_dict(self) -> list[dict[str, Any]]:
        messages_dict = []
        for msg in self._messages:
            d: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.reasoning_content is not None:
                d["reasoning_content"] = msg.reasoning_content
            if msg.tool_calls is not None:
                d["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            messages_dict.append(d)
        return messages_dict

    def _format_fallback(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        if self._template_str and "<|im_start|>" in self._template_str:
            return self._format_chatml(messages, add_generation_prompt)
        if self._template_str and "<|start_header_id|>" in self._template_str:
            return self._format_llama3(messages, add_generation_prompt)

        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"[{role.title()}]: {content}")

        result = "\n".join(parts)
        if add_generation_prompt:
            result += "\n[Assistant]:"
        return result

    def _format_chatml(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "tool":
                parts.append(
                    f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
                )
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        result = "".join(parts)
        if add_generation_prompt:
            result += "<|im_start|>assistant\n"
        return result

    def _format_llama3(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            header = "ipython" if role == "tool" else role
            parts.append(
                f"<|start_header_id|>{header}<|end_header_id|>\n\n{content}<|eot_id|>"
            )
        result = "".join(parts)
        if add_generation_prompt:
            result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return result
