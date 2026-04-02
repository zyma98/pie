"""Models interface implementation -- Model and Tokenizer resources."""

from wit_world.exports.models import Model as ModelBase, Tokenizer as TokenizerBase
from wit_world.imports import runtime as _runtime
from wit_world.imports import tokenize as _tokenize


class Tokenizer(TokenizerBase):
    def __init__(self, host_tokenizer):
        self._inner = host_tokenizer

    def tokenize(self, text: str) -> list[int]:
        return list(self._inner.tokenize(text))

    def detokenize(self, tokens: list[int]) -> str:
        return self._inner.detokenize(tokens)

    def get_vocabs(self) -> tuple[list[int], list[bytes]]:
        return self._inner.get_vocabs()

    def get_special_tokens(self) -> tuple[list[int], list[bytes]]:
        return self._inner.get_special_tokens()

    def get_split_regex(self) -> str:
        return self._inner.get_split_regex()


class Model(ModelBase):
    def __init__(self, host_model):
        self._inner = host_model

    @classmethod
    def get_by_name(cls, name: str):
        host_model = _runtime.get_model(name)
        if host_model is None:
            return None
        return cls(host_model)

    @classmethod
    def get_auto(cls):
        names = list(_runtime.get_all_models())
        if not names:
            raise ValueError("No models available")
        host_model = _runtime.get_model(names[0])
        if host_model is None:
            raise ValueError(f"Model {names[0]} not found")
        return cls(host_model)

    @classmethod
    def get_all_names(cls) -> list[str]:
        return list(_runtime.get_all_models())

    def get_name(self) -> str:
        return self._inner.get_name()

    def get_traits(self) -> list[str]:
        return list(self._inner.get_traits())

    def has_traits(self, required_traits: list[str]) -> bool:
        available = set(self.get_traits())
        return all(t in available for t in required_traits)

    def get_description(self) -> str:
        return self._inner.get_description()

    def get_prompt_template(self) -> str:
        return self._inner.get_prompt_template()

    def eos_tokens(self) -> list[list[int]]:
        tokenizer = _tokenize.get_tokenizer(self._inner)
        return [
            list(tokenizer.tokenize(stop_token))
            for stop_token in self._inner.get_stop_tokens()
        ]

    def get_service_id(self) -> int:
        return self._inner.get_service_id()

    def get_kv_page_size(self) -> int:
        return self._inner.get_kv_page_size()

    def get_tokenizer(self):
        host_tokenizer = _tokenize.get_tokenizer(self._inner)
        return Tokenizer(host_tokenizer)
