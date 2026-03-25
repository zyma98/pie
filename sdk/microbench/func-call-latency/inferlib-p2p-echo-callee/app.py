from wit_world import exports


class Echo(exports.Echo):
    def echo(self, s: str) -> str:
        return s
