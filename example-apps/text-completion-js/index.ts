// Text Completion Example - JavaScript/TypeScript Inferlet
// Demonstrates basic text generation using the inferlet-js library

import {
  getAutoModel,
  getArguments,
  getInstanceId,
  getVersion,
  send,
  Context,
  Sampler,
  maxLen,
  endsWithAny,
} from 'inferlet-js';

const HELP = `
Usage: text-completion-js [OPTIONS]

A simple text completion example for the Pie runtime.

Options:
  --prompt=<text>      The prompt to complete (default: "Hello, world!")
  --max-tokens=<n>     Maximum tokens to generate (default: 256)
  --system=<text>      System prompt (default: helpful assistant)
  -h, --help           Prints this help message
`;

// Parse command line arguments
function parseArgs(args: string[]): {
  help: boolean;
  prompt: string;
  maxTokens: number;
  system: string;
} {
  let help = false;
  let prompt = 'Hello, world!';
  let maxTokens = 256;
  let system = 'You are a helpful, respectful and honest assistant.';

  for (const arg of args) {
    if (arg === '-h' || arg === '--help') {
      help = true;
    } else if (arg.startsWith('--prompt=')) {
      prompt = arg.slice('--prompt='.length);
    } else if (arg.startsWith('--max-tokens=')) {
      maxTokens = parseInt(arg.slice('--max-tokens='.length), 10);
    } else if (arg.startsWith('--system=')) {
      system = arg.slice('--system='.length);
    }
  }

  return { help, prompt, maxTokens, system };
}

// Main implementation
async function main(): Promise<void> {
  const args = getArguments();
  const { help, prompt, maxTokens, system } = parseArgs(args);

  if (help) {
    send(HELP);
    return;
  }

  // Display runtime info
  const instanceId = getInstanceId();
  const version = getVersion();
  send(`Instance: ${instanceId}, Runtime: ${version}\n`);

  // Get the model
  const model = getAutoModel();
  send(`Using model: ${model.getName()}\n`);

  // Create a context for generation
  const ctx = new Context(model);

  // Fill the conversation
  ctx.fillSystem(system);
  ctx.fillUser(prompt);

  send(`\nGenerating response...\n\n`);

  // Create sampler and stop condition
  const sampler = Sampler.topP(0.6, 0.95);
  const eosTokens = model.eosTokens().map((arr) => [...arr]);
  const stopCond = maxLen(maxTokens).or(endsWithAny(eosTokens));

  // Generate the response
  const result = await ctx.generate(sampler, stopCond);

  // Send the result
  send(result);
  send('\n');
}

// Export in WIT-compatible format for inferlet:core/run interface
export const run = {
  run: async (): Promise<{ tag: 'ok' } | { tag: 'err'; val: string }> => {
    try {
      await main();
      return { tag: 'ok' };
    } catch (e) {
      return { tag: 'err', val: String(e) };
    }
  },
};
