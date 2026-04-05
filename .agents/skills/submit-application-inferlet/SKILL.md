---
name: submit-application-inferlet
description: How to submit an application inferlet to the engine server
---
# Submit Application Inferlet
Detailed instructions for the agent on how to submit an application inferlet to the engine server.

## When to Use
- Use this skill when the user asks you to submit an application inferlet to the engine server.
- This skill is helpful for users who want to run an application inferlet.

## Instructions
- Refer to the `engine-server` skill for how to start the engine server and later stop it when you are done. The server should be started before submitting the application inferlet.
- Refer to the `bulid-rust-inferlet` and `build-python-inferlet` skills for more details on how to build the inferlet.
- Once the application inferlet is built as a Wasm binary, you can submit it using the `pie-client` command.
- To use the `pie-client` command, you need to first `source ./pie/.venv/bin/activate` to activate the Pie Python environment.
- Then, use command `pie-client submit --path <application-wasm-path> --manifest <application-manifest-path> -- [additional-arguments]` to submit the application inferlet.
- The manifest file is the `Pie.toml` file in the application source directory. If you see a "dependencies" section in the manifest file, you may need to refer to the `install-dependency-inferlet` skill for how to install the dependencies first before submitting the application inferlet.
- The additional arguments are the arguments to pass to the application inferlet.
- You should get a success message if the application inferlet is submitted successfully, otherwise you will see an error message.
- You should get the inferlet output following the success message.
