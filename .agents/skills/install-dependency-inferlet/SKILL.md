---
name: install-dependency-inferlet
description: How to install dependencies for an inferlet
---
# Install Dependency Inferlet
Detailed instructions for the agent on how to install dependencies for an inferlet.

## When to Use
- Use this skill when the user asks you to install dependencies for an inferlet.
- This skill is helpful for users who want to use an inferlet that requires dependencies.

## Instructions
- Refer to the `engine-server` skill for how to start the engine server and later stop it when you are done. The server should be started before installing the dependency.
- Change directory to the dependency source directory. 
- For now, all dependencies are stored in the `sdk/rust/inferlib` directory.
- Refer to the `bulid-rust-inferlet` and `build-python-inferlet` skills for more details on how to build the inferlet.
- Once the dependency is built as a Wasm binary, you can install it using the `pie-client` command.
- To use the `pie-client` command, you need to first `source ./pie/.venv/bin/activate` to activate the Pie Python environment.
- Then, use command `pie-client install --path <dependency-wasm-path> --manifest <dependency-manifest-path>` to install the dependency.
- The manifest file is the `Pie.toml` file in the dependency source directory.
- You should see a success message if the dependency is installed successfully.
- The engine server may cache previously installed dependencies, so you may see a message saying that the dependency is already installed, which also means success.
