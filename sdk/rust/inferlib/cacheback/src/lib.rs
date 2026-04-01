wit_bindgen::generate!({
    path: "wit",
    world: "cacheback-provider",
    generate_all,
});

use exports::inferlib::cacheback::cacheback::{DraftResult, Guest, GuestCacheTable};
use indexmap::IndexMap;
use std::cell::RefCell;

struct Component;

export!(Component);

impl Guest for Component {
    type CacheTable = CacheTableImpl;
}

pub struct CacheTableImpl {
    inner: RefCell<TwoLevelLruCache>,
    prev_window: RefCell<Vec<u32>>,
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

struct TrieNode {
    children: Vec<TrieNode>,
    token: u32,
    position: u32,
}

impl TrieNode {
    fn new(position: u32, token: u32) -> Self {
        Self {
            children: Vec::new(),
            token,
            position,
        }
    }
}

/// A Trie forest for organizing speculative token drafts into a tree structure.
struct TrieForest {
    roots: Vec<TrieNode>,
    root_position: u32,
}

impl TrieForest {
    fn new(root_position: u32) -> Self {
        Self {
            roots: Vec::new(),
            root_position,
        }
    }

    fn insert(&mut self, tokens: &[u32], positions: &[u32]) {
        if tokens.is_empty() || positions.is_empty() {
            return;
        }

        if positions[0] != self.root_position {
            return;
        }

        let mut candidate_nodes = &mut self.roots;

        for (&token, &position) in tokens.iter().zip(positions.iter()) {
            let candidate_node = candidate_nodes
                .iter()
                .find(|node| node.position == position && node.token == token);

            if candidate_node.is_none() {
                candidate_nodes.push(TrieNode::new(position, token));
            }

            let next_node = candidate_nodes
                .iter_mut()
                .find(|node| node.position == position && node.token == token)
                .unwrap();

            candidate_nodes = &mut next_node.children;
        }
    }

    fn linearize(&self) -> (Vec<u32>, Vec<u32>) {
        fn dfs(node: &TrieNode, tokens: &mut Vec<u32>, positions: &mut Vec<u32>) {
            tokens.push(node.token);
            positions.push(node.position);
            for child in node.children.iter() {
                dfs(child, tokens, positions);
            }
        }

        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        for node in self.roots.iter() {
            dfs(node, &mut tokens, &mut positions);
        }
        (tokens, positions)
    }
}

impl GuestCacheTable for CacheTableImpl {
    fn new(
        leader_capacity: u32,
        follower_capacity: u32,
        leader_len: u32,
        follower_len: u32,
    ) -> Self {
        let window_len = leader_len as usize + follower_len as usize - 1;
        Self {
            inner: RefCell::new(TwoLevelLruCache::new(
                leader_capacity as usize,
                follower_capacity as usize,
                leader_len as usize,
                follower_len as usize,
            )),
            prev_window: RefCell::new(vec![0u32; window_len]),
        }
    }

    fn update(&self, context: Vec<u32>) {
        let mut cache = self.inner.borrow_mut();
        let mut prev_window = self.prev_window.borrow_mut();

        let total = cache.leader_len + cache.follower_len;
        let window_len = total - 1;

        let mut full = Vec::with_capacity(prev_window.len() + context.len());
        full.extend_from_slice(&prev_window);
        full.extend_from_slice(&context);

        if full.len() >= total {
            for i in 0..=full.len() - total {
                let leader = full[i..i + cache.leader_len].to_vec();
                let follower = full[i + cache.leader_len..i + total].to_vec();
                cache.put(leader, follower);
            }
        }

        *prev_window = full[full.len() - window_len..].to_vec();
    }

    fn draft(&self) -> DraftResult {
        let mut cache = self.inner.borrow_mut();
        let prev_window = self.prev_window.borrow();

        let leader_len = cache.leader_len;
        let follower_len = cache.follower_len;
        let positions: Vec<u32> = (1..=follower_len as u32).collect();
        let mut trie = TrieForest::new(1);

        let key = &prev_window[prev_window.len() - leader_len..];
        if let Some(drafts) = cache.get(key) {
            for d in &drafts {
                trie.insert(d, &positions);
            }
        }

        let (tokens, pos) = trie.linearize();
        DraftResult {
            tokens,
            positions: pos,
        }
    }

    fn clear(&self) {
        self.inner.borrow_mut().clear();
        let mut prev_window = self.prev_window.borrow_mut();
        for v in prev_window.iter_mut() {
            *v = 0;
        }
    }
}
