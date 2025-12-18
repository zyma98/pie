// inferlet-js/vitest.config.ts
import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

export default defineConfig({
  test: {
    include: ['src/__tests__/**/*.test.ts', 'test/wasm/__tests__/**/*.test.ts'],
    environment: 'node',
    alias: {
      // Map WIT imports to mock files for unit tests
      'inferlet:core/runtime': resolve(__dirname, 'test/mocks/inferlet-core-runtime.js'),
      'inferlet:core/common': resolve(__dirname, 'test/mocks/inferlet-core-common.js'),
      'inferlet:core/forward': resolve(__dirname, 'test/mocks/inferlet-core-forward.js'),
      'inferlet:core/tokenize': resolve(__dirname, 'test/mocks/inferlet-core-tokenize.js'),
      'inferlet:core/kvs': resolve(__dirname, 'test/mocks/inferlet-core-kvs.js'),
      'inferlet:core/message': resolve(__dirname, 'test/mocks/inferlet-core-message.js'),
      'wasi:io/poll': resolve(__dirname, 'test/mocks/wasi-io-poll.js'),
    },
  },
});
