use indexmap::IndexMap;
use inferlet::drafter::Drafter;

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
/// See [`inferlet::drafter::Drafter`] for the expected shape.
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

/// A cache-based drafter that records token n-gram patterns and speculates future
/// tokens using a two-level LRU cache with O(1) amortized operations.
pub struct CacheDrafter {
    cache: TwoLevelLruCache,
    prev_window: Vec<u32>,
}

impl CacheDrafter {
    pub fn new(
        leader_capacity: usize,
        follower_capacity: usize,
        leader_len: usize,
        follower_len: usize,
    ) -> Self {
        Self {
            cache: TwoLevelLruCache::new(
                leader_capacity,
                follower_capacity,
                leader_len,
                follower_len,
            ),
            prev_window: vec![0; leader_len + follower_len - 1],
        }
    }
}

impl Drafter for CacheDrafter {
    fn update(&mut self, context: &[u32]) {
        let total = self.cache.leader_len + self.cache.follower_len;
        let window_len = total - 1;

        let mut full = Vec::with_capacity(self.prev_window.len() + context.len());
        full.extend_from_slice(&self.prev_window);
        full.extend_from_slice(context);

        if full.len() >= total {
            for i in 0..=full.len() - total {
                let leader = full[i..i + self.cache.leader_len].to_vec();
                let follower = full[i + self.cache.leader_len..i + total].to_vec();
                self.cache.put(leader, follower);
            }
        }

        self.prev_window = full[full.len() - window_len..].to_vec();
    }

    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        let leader_len = self.cache.leader_len;
        let follower_len = self.cache.follower_len;
        let positions: Vec<u32> = (1..=follower_len as u32).collect();
        let mut trie = TrieForest::new(1);

        let key = &self.prev_window[self.prev_window.len() - leader_len..];
        if let Some(drafts) = self.cache.get(key) {
            for d in &drafts {
                trie.insert(d, &positions);
            }
        }

        trie.linearize()
    }
}
