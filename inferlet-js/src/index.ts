// inferlet-js - JavaScript library for writing Pie inferlets
// This mirrors the Rust inferlet crate API

export const VERSION = '0.1.0';

// Re-export runtime functions
export {
  getVersion,
  getInstanceId,
  getArguments,
  setReturn,
  debugQuery,
} from './runtime.js';

// Re-export Sampler
export { Sampler } from './sampler.js';
export type { SamplerType } from './sampler.js';
