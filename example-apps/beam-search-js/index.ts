// Beam Search Example - JavaScript/TypeScript Inferlet
// Demonstrates beam search decoding for text generation using the inferlet-js library

import {
  getAutoModel,
  getArguments,
  getInstanceId,
  getVersion,
  send,
  Context,
  maxLen,
  endsWithAny,
} from 'inferlet-js';

const HELP = `
Usage: beam-search-js [OPTIONS]

A program to run text generation with beam search decoding.

Options:
  --prompt=<text>      The prompt to send to the model
                       [default: Explain the LLM decoding process ELI5.]
  --max-tokens=<n>     Maximum tokens to generate (default: 128)
  --beam-size=<n>      The beam size for decoding (default: 1)
  --system=<text>      System prompt (default: helpful assistant)
  -h, --help           Prints this help message
`;

// Parse command line arguments
function parseArgs(args: string[]): {
  help: boolean;
  prompt: string;
  maxTokens: number;
  beamSize: number;
  system: string;
} {
  let help = false;
  let prompt = 'Explain the LLM decoding process ELI5.';
  let maxTokens = 128;
  let beamSize = 1;
  let system = 'You are a helpful, respectful and honest assistant.';

  for (const arg of args) {
    if (arg === '-h' || arg === '--help') {
      help = true;
    } else if (arg.startsWith('--prompt=')) {
      prompt = arg.slice('--prompt='.length);
    } else if (arg.startsWith('--max-tokens=')) {
      maxTokens = parseInt(arg.slice('--max-tokens='.length), 10);
    } else if (arg.startsWith('--beam-size=')) {
      beamSize = parseInt(arg.slice('--beam-size='.length), 10);
    } else if (arg.startsWith('--system=')) {
      system = arg.slice('--system='.length);
    }
  }

  return { help, prompt, maxTokens, beamSize, system };
}

// Main implementation
async function main(): Promise<void> {
  const args = getArguments();
  const { help, prompt, maxTokens, beamSize, system } = parseArgs(args);

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

  send(`\nGenerating response with beam search (beam size: ${beamSize})...\n\n`);

  // Create stop condition
  const eosTokens = model.eosTokens().map((arr) => [...arr]);
  const stopCond = maxLen(maxTokens).or(endsWithAny(eosTokens));

  // Measure generation time
  const startTime = Date.now();

  // Generate the response using beam search
  const result = await ctx.generateWithBeam(stopCond, beamSize);

  const elapsedTime = Date.now() - startTime;

  // Send the result
  send(result);
  send('\n');

  // Calculate and display performance metrics
  const tokenIds = model.getTokenizer().tokenize(result);
  send(`\nTotal elapsed: ${elapsedTime}ms\n`);

  if (tokenIds.length > 0) {
    const perTokenLatency = elapsedTime / tokenIds.length;
    send(`Per token latency: ${perTokenLatency.toFixed(2)}ms\n`);
  }
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
