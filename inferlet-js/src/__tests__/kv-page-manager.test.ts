import { describe, it, expect, vi, beforeEach } from 'vitest';
import { KvPageManager } from '../kv-page-manager.js';

// Mock the Queue's newKvPages method
const mockNewKvPages = vi.fn();
const mockQueue = {
  newKvPages: mockNewKvPages,
} as any;

describe('KvPageManager', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should start with zero pages', () => {
    const manager = new KvPageManager(mockQueue, 128);
    expect(manager.pageCount).toBe(0);
    expect(manager.lastPageLen).toBe(0);
    expect(manager.totalTokens).toBe(0);
  });

  it('should allocate pages on grow', () => {
    const mockPage = { ptr: 1, ref: vi.fn(), release: vi.fn() };
    mockNewKvPages.mockReturnValue([mockPage]);

    const manager = new KvPageManager(mockQueue, 128);
    manager.grow(100);

    expect(mockNewKvPages).toHaveBeenCalledWith(1);
    expect(manager.pageCount).toBe(1);
    expect(manager.lastPageLen).toBe(100);
  });

  it('should allocate multiple pages for large grows', () => {
    const mockPages = [
      { ptr: 1, ref: vi.fn(), release: vi.fn() },
      { ptr: 2, ref: vi.fn(), release: vi.fn() },
    ];
    mockNewKvPages.mockReturnValue(mockPages);

    const manager = new KvPageManager(mockQueue, 128);
    manager.grow(200);

    expect(mockNewKvPages).toHaveBeenCalledWith(2);
    expect(manager.pageCount).toBe(2);
    expect(manager.lastPageLen).toBe(72); // 200 % 128 = 72
  });

  it('should shrink pages when tokens are removed', () => {
    const mockPages = [
      { ptr: 1, ref: vi.fn(), release: vi.fn() },
      { ptr: 2, ref: vi.fn(), release: vi.fn() },
    ];
    mockNewKvPages.mockReturnValue(mockPages);

    const manager = new KvPageManager(mockQueue, 128);
    manager.grow(200); // 2 pages, lastPageLen = 72

    expect(manager.pageCount).toBe(2);
    expect(manager.totalTokens).toBe(200);

    // Shrink by 100 tokens -> 100 tokens remaining -> 1 page needed
    manager.shrink(100);

    expect(manager.pageCount).toBe(1);
    expect(manager.totalTokens).toBe(100);
    expect(manager.lastPageLen).toBe(100);
    // Second page should have been released
    expect(mockPages[1].release).toHaveBeenCalled();
  });

  it('should handle shrink to zero', () => {
    const mockPage = { ptr: 1, ref: vi.fn(), release: vi.fn() };
    mockNewKvPages.mockReturnValue([mockPage]);

    const manager = new KvPageManager(mockQueue, 128);
    manager.grow(50);

    expect(manager.pageCount).toBe(1);

    manager.shrink(50);

    expect(manager.pageCount).toBe(0);
    expect(manager.totalTokens).toBe(0);
    expect(mockPage.release).toHaveBeenCalled();
  });

  it('should throw on underflow', () => {
    const manager = new KvPageManager(mockQueue, 128);

    expect(() => manager.shrink(10)).toThrow('underflow');
  });
});
