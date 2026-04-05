---
name: build-python-inferlet
description: How to build a Python inferlet
---
# Build Python Inferlet
Detailed instructions for the agent on how to build a Python inferlet.

## When to Use
- Use this skill when the user asks you to build a Python inferlet.
- This skill is helpful for users who want to build a Python inferlet.

## Instructions
- Change directory to the Python inferlet source directory.
- Activate the Python environment by running `source ./pie/.venv/bin/activate`.
- Let's assume that the repository root is `$REPO`.
- If the inferlet is an application, you can build it by running `bakery build --inferlib "$REPO/sdk/rust/inferlib" <inferlet-source-directory> -o <output.wasm>` to build.
- If the inferlet is a library, you can build it by running `bakery build --lib --inferlib "$REPO/sdk/rust/inferlib" --world <world-name> <inferlet-source-directory> -o <output.wasm>` to build. You can find the world name in the `wit/` directory of the inferlet source directory.
- The output will generate a `shared/` and a `runtime/` directory next to the output Wasm binary. You can ignore these two directories.
