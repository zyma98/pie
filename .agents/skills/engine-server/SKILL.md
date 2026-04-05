---
name: engine-server
description: How to start and stop the Pie engine server
---
# Engine Server
Detailed instructions for the agent on how to start and stop the Pie engine server.

## When to Use
- Use this skill when the user asks you to start or stop the Pie engine server.
- This skill is helpful for users who want to use the Pie engine server to run inferlets.

## Instructions
- Make sure the Pie engine server is not already running. You can run `ps aux | grep "uv run pie serve"` to check.
- Change directory to the `pie/` subdirectory in this repository.
- Run `uv run pie serve` to start the Pie engine server. Remember to record the process ID of the server for later stopping it. Please note that you should record the process ID returned by the `uv` command.
- The server will become available after you see "Engine running" in the output.
- To stop the server, use `kill -TERM <process-id>` to send a termination signal to the server process.
- To double check, you can run `ps aux | grep "uv run pie serve"` to make sure the server process is no longer running.
