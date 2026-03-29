from wit_world import exports

class Echo(exports.Echo):
    def echo(self, s: str) -> str:
        return s

def main() -> None:
    print("Hello, world!")


if __name__ == "__main__":
    main()
