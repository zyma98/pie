use std::collections::BTreeSet;

/// Binary Run-Length Encoding (BRLE) structure.
///
/// Efficiently stores a large sequence of booleans by encoding
/// the lengths of consecutive runs of `false` and `true` values.
/// The sequence always begins with a run of `false`s, which may be zero-length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Brle {
    buffer: Vec<u32>,
    total_size: usize,
}

impl Brle {
    pub fn new(size: usize) -> Self {
        if size == 0 {
            Self {
                buffer: vec![],
                total_size: 0,
            }
        } else {
            Self {
                buffer: vec![size as u32],
                total_size: size,
            }
        }
    }

    pub fn get_buffer(&self) -> Vec<u32> {
        self.buffer.clone()
    }

    pub fn append(&mut self, flag: bool) {
        if self.buffer.is_empty() {
            if flag {
                self.buffer.extend(&[0, 1]);
            } else {
                self.buffer.push(1);
            }
        } else {
            let last_run_is_true = (self.buffer.len() - 1) % 2 != 0;
            if last_run_is_true == flag {
                *self.buffer.last_mut().unwrap() += 1;
            } else {
                self.buffer.push(1);
            }
        }
        self.total_size += 1;
    }

    fn iter_runs(&self) -> RunIterator<'_> {
        RunIterator {
            buffer: &self.buffer,
            index: 0,
            current_pos: 0,
        }
    }

    fn slice(&self, start: usize, end: usize) -> Self {
        let end = end.min(self.total_size);
        if start >= end {
            return Self::new(0);
        }

        let new_size = end - start;
        let mut new_buffer = Vec::new();

        for (val, r_start, r_end) in self.iter_runs() {
            let slice_r_start = r_start.max(start);
            let slice_r_end = r_end.min(end);

            if slice_r_start < slice_r_end {
                let len = (slice_r_end - slice_r_start) as u32;

                if new_buffer.is_empty() {
                    if val {
                        new_buffer.push(0);
                    }
                    new_buffer.push(len);
                } else {
                    let last_run_is_true = (new_buffer.len() - 1) % 2 != 0;
                    if last_run_is_true == val {
                        *new_buffer.last_mut().unwrap() += len;
                    } else {
                        new_buffer.push(len);
                    }
                }
            }
        }

        Self {
            buffer: new_buffer,
            total_size: new_size,
        }
    }

    fn extend(&mut self, other: &Self) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = other.clone();
            return;
        }

        let self_last_run_is_true = (self.buffer.len() - 1) % 2 != 0;
        let other_first_run_is_true = other.buffer.first() == Some(&0) && other.buffer.len() > 1;

        if self_last_run_is_true == other_first_run_is_true {
            let other_first_run_len = if other_first_run_is_true {
                other.buffer[1]
            } else {
                other.buffer[0]
            };
            let other_slice_start = if other_first_run_is_true { 2 } else { 1 };

            *self.buffer.last_mut().unwrap() += other_first_run_len;
            self.buffer
                .extend_from_slice(&other.buffer[other_slice_start..]);
        } else if other_first_run_is_true {
            self.buffer.extend_from_slice(&other.buffer[1..]);
        } else {
            self.buffer.extend_from_slice(&other.buffer);
        }
        self.total_size += other.total_size;
    }

    pub fn len(&self) -> usize {
        self.total_size
    }

    pub fn is_empty(&self) -> bool {
        self.total_size == 0
    }

    pub fn to_bools(&self) -> Vec<bool> {
        let mut vec = Vec::with_capacity(self.total_size);
        for (value, start, end) in self.iter_runs() {
            let run_len = end - start;
            for _ in 0..run_len {
                vec.push(value);
            }
        }
        vec
    }

    pub fn from_bools(v: &[bool]) -> Self {
        if v.is_empty() {
            return Self::new(0);
        }

        let mut buffer = Vec::new();
        let mut current_val = false;
        let mut count = 0;

        if v[0] {
            buffer.push(0);
            current_val = true;
        }

        for &val in v {
            if val == current_val {
                count += 1;
            } else {
                buffer.push(count);
                current_val = val;
                count = 1;
            }
        }
        buffer.push(count);

        Self {
            buffer,
            total_size: v.len(),
        }
    }

    pub fn mask_range(&mut self, start: usize, end: usize, flag: bool) {
        if start >= end {
            return;
        }
        let ranges = vec![(start, end)];
        self.mask_internal(&ranges, flag);
    }

    pub fn mask(&mut self, indices: &[usize], flag: bool) {
        if indices.is_empty() {
            return;
        }

        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable();
        sorted_indices.dedup();

        let mut ranges = Vec::new();
        if sorted_indices.is_empty() {
            return;
        }

        let mut range_start = sorted_indices[0];
        let mut range_end = range_start + 1;

        for &index in sorted_indices.iter().skip(1) {
            if index == range_end {
                range_end = index + 1;
            } else {
                ranges.push((range_start, range_end));
                range_start = index;
                range_end = index + 1;
            }
        }
        ranges.push((range_start, range_end));

        self.mask_internal(&ranges, flag);
    }

    pub fn remove_range(&mut self, start: usize, end: usize) {
        let end = end.min(self.total_size);
        if start >= end {
            return;
        }

        let head = self.slice(0, start);
        let tail = self.slice(end, self.total_size);

        let mut new_brle = head;
        new_brle.extend(&tail);
        *self = new_brle;
    }

    pub fn is_range_all_value(&self, start: usize, end: usize, expected_value: bool) -> bool {
        if start >= end {
            return true;
        }
        if end > self.total_size {
            return false;
        }

        let mut pos_covered = start;

        for (run_value, run_start, run_end) in self.iter_runs() {
            let intersect_start = run_start.max(pos_covered);
            let intersect_end = run_end.min(end);

            if intersect_start < intersect_end {
                if run_value != expected_value {
                    return false;
                }
                pos_covered = intersect_end;

                if pos_covered >= end {
                    return true;
                }
            }

            if run_end >= end {
                break;
            }
        }

        pos_covered >= end
    }

    fn mask_internal(&mut self, ranges: &[(usize, usize)], flag: bool) {
        if ranges.is_empty() || self.total_size == 0 {
            return;
        }

        let mut events = BTreeSet::new();
        events.insert(0);
        events.insert(self.total_size);

        for &(start, end) in ranges {
            let clamped_start = start.min(self.total_size);
            let clamped_end = end.min(self.total_size);
            if clamped_start < clamped_end {
                events.insert(clamped_start);
                events.insert(clamped_end);
            }
        }

        for run in self.iter_runs() {
            events.insert(run.1);
            events.insert(run.2);
        }

        let runs: Vec<_> = self.iter_runs().collect();
        let mut run_idx = 0;

        let mut new_buffer = Vec::new();
        let mut range_iter = ranges.iter().peekable();

        let event_points: Vec<_> = events.into_iter().collect();
        for window in event_points.windows(2) {
            let start = window[0];
            let end = window[1];
            if start >= end {
                continue;
            }

            let mid_point = start + (end - start) / 2;

            let is_masked = loop {
                match range_iter.peek() {
                    Some(&&(r_start, r_end)) => {
                        if mid_point >= r_end {
                            range_iter.next();
                            continue;
                        }
                        break mid_point >= r_start && mid_point < r_end;
                    }
                    None => break false,
                }
            };

            let value = if is_masked {
                flag
            } else {
                while run_idx < runs.len() && mid_point >= runs[run_idx].2 {
                    run_idx += 1;
                }
                runs.get(run_idx)
                    .expect("Should always find a run for a valid midpoint")
                    .0
            };

            let len = (end - start) as u32;

            let should_merge = if new_buffer.last().is_some() {
                let last_val_is_true = (new_buffer.len() - 1) % 2 != 0;
                last_val_is_true == value
            } else {
                false
            };

            if should_merge {
                *new_buffer.last_mut().unwrap() += len;
            } else {
                if new_buffer.is_empty() && value {
                    new_buffer.push(0);
                }
                new_buffer.push(len);
            }
        }
        self.buffer = new_buffer;
    }
}

struct RunIterator<'a> {
    buffer: &'a [u32],
    index: usize,
    current_pos: usize,
}

impl<'a> Iterator for RunIterator<'a> {
    type Item = (bool, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.len() {
            return None;
        }

        let run_len = self.buffer[self.index] as usize;
        let value = self.index % 2 != 0;

        let start = self.current_pos;
        let end = self.current_pos + run_len;

        self.current_pos = end;
        self.index += 1;

        Some((value, start, end))
    }
}
