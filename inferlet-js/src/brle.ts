// Binary Run-Length Encoding (BRLE) structure.
// This efficiently stores a large sequence of booleans by encoding
// the lengths of consecutive runs of `false` and `true` values.
// Mirrors the Rust Brle from inferlet/src/brle.rs

/**
 * A Binary Run-Length Encoding (BRLE) structure.
 *
 * This structure efficiently stores a large sequence of booleans by encoding
 * the lengths of consecutive runs of `false` and `true` values.
 * The sequence always begins with a run of `false`s, which may be zero-length.
 *
 * Encoding examples:
 * - `[false, false, true, true, true, false]` is encoded as `[2, 3, 1]`.
 * - `[true, true, false]` is encoded as `[0, 2, 1]`.
 */
export class Brle {
  /** The buffer of run lengths. Even indices are for `false` runs, odd for `true`. */
  buffer: number[];
  /** The total number of boolean values represented. */
  totalSize: number;

  private constructor(buffer: number[], totalSize: number) {
    this.buffer = buffer;
    this.totalSize = totalSize;
  }

  /**
   * Creates a new `Brle` instance representing `size` `true` values.
   * This is the default causal mask behavior (all visible).
   */
  static new(size: number): Brle {
    if (size === 0) {
      return new Brle([], 0);
    }
    // All true values: [0, size] means 0 false values, then size true values
    return new Brle([0, size], size);
  }

  /**
   * Creates a `Brle` from an array of booleans.
   */
  static fromArray(values: boolean[]): Brle {
    if (values.length === 0) {
      return Brle.new(0);
    }

    const buffer: number[] = [];
    let currentVal = false;
    let count = 0;

    // Handle the initial run of `false`s, which could be zero-length.
    if (values[0]) {
      buffer.push(0);
      currentVal = true;
    }

    for (const val of values) {
      if (val === currentVal) {
        count++;
      } else {
        buffer.push(count);
        currentVal = val;
        count = 1;
      }
    }
    buffer.push(count); // Push the last run

    return new Brle(buffer, values.length);
  }

  /**
   * Returns the total number of booleans in the sequence.
   */
  len(): number {
    return this.totalSize;
  }

  /**
   * Returns `true` if the sequence is empty.
   */
  isEmpty(): boolean {
    return this.totalSize === 0;
  }

  /**
   * Decodes the `Brle` into an array of booleans.
   */
  toArray(): boolean[] {
    const result: boolean[] = [];
    for (const [value, start, end] of this.iterRuns()) {
      const runLen = end - start;
      for (let i = 0; i < runLen; i++) {
        result.push(value);
      }
    }
    return result;
  }

  /**
   * Appends a boolean value to the end of the sequence.
   */
  append(flag: boolean): void {
    if (this.buffer.length === 0) {
      if (flag) {
        this.buffer.push(0, 1);
      } else {
        this.buffer.push(1);
      }
    } else {
      const lastRunIsTrue = (this.buffer.length - 1) % 2 !== 0;
      if (lastRunIsTrue === flag) {
        this.buffer[this.buffer.length - 1]++;
      } else {
        this.buffer.push(1);
      }
    }
    this.totalSize++;
  }

  /**
   * Sets a range of booleans to a specified value.
   * The range is exclusive of `end` (`start..end`).
   */
  maskRange(start: number, end: number, flag: boolean): void {
    if (start >= end) {
      return;
    }
    this.maskInternal([[start, end]], flag);
  }

  /**
   * Sets multiple indices to a specified value.
   */
  mask(indices: number[], flag: boolean): void {
    if (indices.length === 0) {
      return;
    }

    // Sort and deduplicate indices
    const sortedIndices = [...new Set(indices)].sort((a, b) => a - b);

    // Group into contiguous ranges
    const ranges: [number, number][] = [];
    let rangeStart = sortedIndices[0];
    let rangeEnd = rangeStart + 1;

    for (let i = 1; i < sortedIndices.length; i++) {
      const index = sortedIndices[i];
      if (index === rangeEnd) {
        rangeEnd = index + 1;
      } else {
        ranges.push([rangeStart, rangeEnd]);
        rangeStart = index;
        rangeEnd = index + 1;
      }
    }
    ranges.push([rangeStart, rangeEnd]);

    this.maskInternal(ranges, flag);
  }

  /**
   * Returns an iterator over the runs, yielding [value, startIndex, endIndex].
   */
  *iterRuns(): Generator<[boolean, number, number]> {
    let currentPos = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const runLen = this.buffer[i];
      const value = i % 2 !== 0;
      const start = currentPos;
      const end = currentPos + runLen;
      currentPos = end;
      yield [value, start, end];
    }
  }

  /**
   * Core masking logic for pre-sorted, disjoint ranges.
   */
  private maskInternal(ranges: [number, number][], flag: boolean): void {
    if (ranges.length === 0 || this.totalSize === 0) {
      return;
    }

    // Collect all event points
    const events = new Set<number>();
    events.add(0);
    events.add(this.totalSize);

    for (const [start, end] of ranges) {
      const clampedStart = Math.min(start, this.totalSize);
      const clampedEnd = Math.min(end, this.totalSize);
      if (clampedStart < clampedEnd) {
        events.add(clampedStart);
        events.add(clampedEnd);
      }
    }

    for (const [, runStart, runEnd] of this.iterRuns()) {
      events.add(runStart);
      events.add(runEnd);
    }

    const eventPoints = [...events].sort((a, b) => a - b);
    const newBuffer: number[] = [];
    const runIterator = this.iterRuns();
    let rangeIndex = 0;
    let currentRun = runIterator.next();

    for (let i = 0; i < eventPoints.length - 1; i++) {
      const start = eventPoints[i];
      const end = eventPoints[i + 1];
      if (start >= end) continue;

      const midPoint = start + Math.floor((end - start) / 2);

      // Check if midPoint is in a masked range
      while (rangeIndex < ranges.length && midPoint >= ranges[rangeIndex][1]) {
        rangeIndex++;
      }
      const isMasked =
        rangeIndex < ranges.length &&
        midPoint >= ranges[rangeIndex][0] &&
        midPoint < ranges[rangeIndex][1];

      let value: boolean;
      if (isMasked) {
        value = flag;
      } else {
        // Find the original value
        while (!currentRun.done && midPoint >= currentRun.value[2]) {
          currentRun = runIterator.next();
        }
        value = currentRun.value![0];
      }

      const len = end - start;

      // Merge with previous run if same value
      const shouldMerge =
        newBuffer.length > 0 &&
        (newBuffer.length - 1) % 2 !== 0 === value;

      if (shouldMerge) {
        newBuffer[newBuffer.length - 1] += len;
      } else {
        if (newBuffer.length === 0 && value) {
          newBuffer.push(0);
        }
        newBuffer.push(len);
      }
    }

    this.buffer = newBuffer;
  }

  /**
   * Creates a clone of this Brle.
   */
  clone(): Brle {
    return new Brle([...this.buffer], this.totalSize);
  }
}

/**
 * Creates a causal attention mask for the given parameters.
 * Each token can only attend to tokens at positions <= its own position.
 *
 * @param numTotalTokens - Total number of tokens in the KV cache
 * @param numInputTokens - Number of new input tokens being processed
 * @returns Array of Brle masks, one per input token
 */
export function causalMask(numTotalTokens: number, numInputTokens: number): Brle[] {
  const masks: Brle[] = [];
  const offset = numTotalTokens - numInputTokens;
  for (let i = 0; i < numInputTokens; i++) {
    masks.push(Brle.new(offset + i + 1));
  }
  return masks;
}
