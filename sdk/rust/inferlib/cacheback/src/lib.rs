wit_bindgen::generate!({
    path: "wit",
    world: "cacheback-provider",
    generate_all,
});

use exports::inferlib::cacheback::cacheback::{Guest, GuestCacheTable};
use indexmap::IndexMap;
use std::cell::RefCell;

struct Component;

export!(Component);

impl Guest for Component {
    type CacheTable = CacheTableImpl;
}

pub struct CacheTableImpl {
    inner: RefCell<TwoLevelLruCache>,
}

/// Two-level LRU cache backed by `IndexMap` for O(1) amortized operations.
///
/// - Leader level: up to `leader_capacity` entries keyed by token windows of
///   length `leader_len`, ordered from LRU (index 0) to MRU (last index).
/// - Follower level: for each leader, up to `follower_capacity` follower
///   token windows of length `follower_len`, with the same LRU ordering.
struct TwoLevelLruCache {
    leader_capacity: usize,
    follower_capacity: usize,
    leader_len: usize,
    follower_len: usize,
    cache: IndexMap<Vec<u32>, IndexMap<Vec<u32>, ()>>,
}

impl TwoLevelLruCache {
    fn new(
        leader_capacity: usize,
        follower_capacity: usize,
        leader_len: usize,
        follower_len: usize,
    ) -> Self {
        assert!(
            leader_capacity > 0 && follower_capacity > 0,
            "capacities must be positive"
        );
        Self {
            leader_capacity,
            follower_capacity,
            leader_len,
            follower_len,
            cache: IndexMap::new(),
        }
    }

    fn touch_leader(&mut self, idx: usize) {
        let last = self.cache.len() - 1;
        self.cache.move_index(idx, last);
    }

    fn put(&mut self, leader: Vec<u32>, follower: Vec<u32>) {
        if let Some(leader_idx) = self.cache.get_index_of(leader.as_slice()) {
            self.touch_leader(leader_idx);
            let last = self.cache.len() - 1;
            let (_, vcache) = self.cache.get_index_mut(last).unwrap();

            if let Some(fi) = vcache.get_index_of(follower.as_slice()) {
                let vlast = vcache.len() - 1;
                vcache.move_index(fi, vlast);
            } else {
                if vcache.len() >= self.follower_capacity {
                    vcache.shift_remove_index(0);
                }
                vcache.insert(follower, ());
            }
        } else {
            if self.cache.len() >= self.leader_capacity {
                self.cache.shift_remove_index(0);
            }
            let mut vcache = IndexMap::new();
            vcache.insert(follower, ());
            self.cache.insert(leader, vcache);
        }
    }

    fn get(&mut self, leader: &[u32]) -> Option<Vec<Vec<u32>>> {
        let leader_idx = self.cache.get_index_of(leader)?;
        self.touch_leader(leader_idx);
        let last = self.cache.len() - 1;
        let (_, vcache) = self.cache.get_index(last).unwrap();
        Some(vcache.keys().cloned().collect())
    }

    fn clear(&mut self) {
        self.cache.clear();
    }
}

impl GuestCacheTable for CacheTableImpl {
    fn new(
        leader_capacity: u32,
        follower_capacity: u32,
        leader_len: u32,
        follower_len: u32,
    ) -> Self {
        Self {
            inner: RefCell::new(TwoLevelLruCache::new(
                leader_capacity as usize,
                follower_capacity as usize,
                leader_len as usize,
                follower_len as usize,
            )),
        }
    }

    fn update_cache(&self, token_ids: Vec<u32>) {
        let mut cache = self.inner.borrow_mut();
        let total = cache.leader_len + cache.follower_len;
        if token_ids.len() < total {
            return;
        }
        for i in 0..=token_ids.len() - total {
            let leader = token_ids[i..i + cache.leader_len].to_vec();
            let follower = token_ids[i + cache.leader_len..i + total].to_vec();
            cache.put(leader, follower);
        }
    }

    fn get_draft_tokens(&self, leader: Vec<u32>) -> Vec<Vec<u32>> {
        let mut cache = self.inner.borrow_mut();
        let leader_len = cache.leader_len;
        let key = if leader.len() >= leader_len {
            &leader[leader.len() - leader_len..]
        } else {
            &leader
        };
        cache.get(key).unwrap_or_default()
    }

    fn clear(&self) {
        self.inner.borrow_mut().clear();
    }
}
