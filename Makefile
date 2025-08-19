.PHONY: example, start, build, client

start:
	cd pie-cli && pie start --config ./example_config.toml

build:
	cd pie-cli && cargo install --path .

example:
	cd example-apps && cargo build --target wasm32-wasip2 --release

client:
	cd client/python && python typego_s0.py