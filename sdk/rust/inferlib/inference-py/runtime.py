"""Runtime interface implementation -- passthrough to host APIs."""

from wit_world.imports import runtime as _runtime


class Runtime:
    def get_version(self) -> str:
        return _runtime.get_version()

    def get_instance_id(self) -> str:
        return _runtime.get_instance_id()

    def get_arguments(self) -> list[str]:
        return list(_runtime.get_arguments())

    def set_return(self, value: str) -> None:
        _runtime.set_return(value)

    def get_all_models_with_traits(self, traits: list[str]) -> list[str]:
        return list(_runtime.get_all_models_with_traits(traits))

    def debug_query(self, query: str) -> str:
        result = _runtime.debug_query(query)
        while True:
            pollable = result.pollable()
            pollable.block()
            value = result.get()
            if value is not None:
                return value
