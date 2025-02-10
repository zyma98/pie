use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::atomic::{AtomicIsize, Ordering};

use std::time::{Duration, Instant};

// ------------------------------------------------------------
// Type aliases matching your Python code
// ------------------------------------------------------------
pub type InstanceId = Vec<u8>;
pub type Address = usize;
pub type BlockPointer = usize;

// ------------------------------------------------------------
// A custom error type to mirror your BlockError in Python
// ------------------------------------------------------------
#[derive(Debug)]
pub enum BlockError {
    NoFreeBlocks,
    NotEnoughFreeBlocks { requested: usize, available: usize },
    VirtualAddressNotFound,
}

// ------------------------------------------------------------
// AddressMap
// ------------------------------------------------------------
#[derive(Debug)]
pub struct AddressMap {
    mapping: HashMap<Address, Address>,
    addr_pool: AddressPool,
}

impl AddressMap {
    pub fn new() -> Self {
        Self {
            mapping: HashMap::new(),
            addr_pool: AddressPool::new(Address::MAX),
        }
    }

    pub fn register(&mut self, addr: Address) -> Address {
        let virtual_addr = self.addr_pool.allocate().unwrap();
        self.mapping.insert(virtual_addr, addr);
        virtual_addr
    }

    pub fn unregister(&mut self, virtual_addr: Address) {
        self.mapping.remove(&virtual_addr);
        self.addr_pool.release(virtual_addr).unwrap();
    }

    pub fn resolve(&self, virtual_addr: Address) -> Result<Address, BlockError> {
        self.mapping
            .get(&virtual_addr)
            .copied()
            .ok_or(BlockError::VirtualAddressNotFound)
    }
}

/// A fast, thread-safe counter.
#[derive(Debug)]
pub struct Counter {
    count: AtomicIsize,
}

impl Counter {
    /// Creates a new counter starting at the given initial value.
    pub fn new(initial: isize) -> Self {
        Self {
            count: AtomicIsize::new(initial),
        }
    }

    /// Increments the counter by 1.
    pub fn inc(&self) {
        // Using relaxed ordering because we only care about the counter's value.
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements the counter by 1.
    pub fn dec(&self) {
        self.count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Returns the current count.
    pub fn get(&self) -> isize {
        self.count.load(Ordering::Relaxed)
    }
}

/// A very fast bounded ID pool that always returns the smallest available ID.
/// The pool is created with a maximum capacity.
#[derive(Debug)]
pub struct AddressPool {
    /// The next (never‑allocated) ID.
    next: Address,
    /// The set of freed IDs.
    free: BTreeSet<Address>,
    /// The maximum number of IDs that can be allocated.
    max_capacity: usize,
}

impl AddressPool {
    /// Create a new ID pool with the given maximum capacity.
    pub fn new(max_capacity: usize) -> Self {
        Self {
            next: 0,
            free: BTreeSet::new(),
            max_capacity,
        }
    }

    /// Allocate and return the smallest available ID.
    ///
    /// Returns `Some(id)` if an ID is available, or `None` if the pool is exhausted.
    pub fn allocate(&mut self) -> Option<Address> {
        if let Some(&id) = self.free.iter().next() {
            // There is a freed ID available. Remove and return it.
            self.free.remove(&id);
            Some(id)
        } else if self.next < self.max_capacity {
            // No freed IDs available; allocate a fresh one.
            let addr = self.next;
            self.next += 1;
            Some(addr)
        } else {
            // Pool is exhausted.
            None
        }
    }

    /// Release an ID back into the pool so it can be re-used.

    pub fn release(&mut self, addr: Address) -> Result<(), BlockError> {
        // Only allow releasing IDs that were allocated.
        if addr >= self.next {
            return Err(BlockError::VirtualAddressNotFound);
        }

        // Insert the id into the free set.
        self.free.insert(addr);

        // Tail optimization: if the largest freed id is exactly next-1,
        // collapse the free block by decrementing `next`.
        while let Some(&last) = self.free.iter().next_back() {
            if last == self.next - 1 {
                self.free.remove(&last);
                self.next -= 1;
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Returns the number of IDs that are available for allocation.
    ///
    /// This equals the number of IDs that have been freed plus the difference
    /// between the maximum capacity and the next never‑allocated ID.
    pub fn available(&self) -> usize {
        self.free.len() + (self.max_capacity - self.next)
    }
}

/// A tracker that records the recency of usage.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecencyTracker {
    last_used: Option<Instant>,
}

impl RecencyTracker {
    /// Creates a new `RecencyTracker` with no recorded usage.
    pub fn new() -> Self {
        Self { last_used: None }
    }

    /// Returns the timestamp of the last use, or `None` if never refreshed.
    pub fn last_used(&self) -> Option<Instant> {
        self.last_used
    }

    /// Updates the tracker to the current time.
    pub fn refresh_last_used(&mut self) {
        self.last_used = Some(Instant::now());
    }
}

// ------------------------------------------------------------
// Basic Storage: Mirroring Python's BlockStorage
// We'll maintain a vector of bool to track if an index is free.
// ------------------------------------------------------------
pub trait BlockStorage {
    fn max_capacity(&self) -> usize;
    fn num_free_blocks(&self) -> usize;

    pub fn new(max_capacity: usize) -> Self {
        Self {
            max_capacity,
            index: vec![true; max_capacity],
        }
    }

    pub fn allocate(&mut self, num_blocks: usize) -> Result<Vec<BlockPointer>, BlockError> {
        // Python code:
        // if self.num_free_blocks() == 0:
        //     raise BlockError("No free blocks available")
        if self.num_free_blocks() == 0 {
            return Err(BlockError::NoFreeBlocks);
        }

        let mut allocated = vec![];
        for (i, &is_free) in self.index.iter().enumerate() {
            if allocated.len() == num_blocks {
                break;
            }
            if is_free {
                allocated.push(i);
            }
        }

        if allocated.len() < num_blocks {
            return Err(BlockError::NotEnoughFreeBlocks {
                requested: num_blocks,
                available: allocated.len(),
            });
        }

        // Mark them used
        for &ptr in &allocated {
            self.index[ptr] = false;
        }

        Ok(allocated)
    }

    pub fn deallocate(&mut self, block_ptrs: &[BlockPointer]) {
        for &ptr in block_ptrs {
            if ptr < self.max_capacity {
                self.index[ptr] = true;
            }
        }
    }

    pub fn num_free_blocks(&self) -> usize {
        self.index.iter().filter(|&&b| b).count()
    }

    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }
}

// ------------------------------------------------------------
// Generic BlockManager
// In Python, it's a generic class: BlockManager[BT: Block, ST: BlockStorage]
// We'll do something similar in Rust with generics.
//
// We'll store a function/closure `create_block_fn` for how to create the block
// from a pointer. This replaces the "abstractmethod create_block" from Python.
// ------------------------------------------------------------
pub struct BlockManager<B, S, F>
where
    B: Block,
    S: StorageTrait, // see below
    F: Fn(BlockPointer) -> B,
{
    // "global" address -> actual block
    blocks: HashMap<Address, B>,

    // track address spaces for each user
    addr_space: HashMap<InstanceId, AddressMap>,

    // The actual storage
    storage: S,

    // optional secondary storage
    storage_secondary: Option<S>,

    // global ID assignment for newly created blocks
    addr_assignment: Address,

    // how to create a block from a pointer
    create_block_fn: F,
}

// We define a "StorageTrait" to unify the minimal methods we need from storage:
pub trait StorageTrait {
    fn allocate(&mut self, num_blocks: usize) -> Result<Vec<BlockPointer>, BlockError>;
    fn deallocate(&mut self, block_ptrs: &[BlockPointer]);
    fn num_free_blocks(&self) -> usize;
}

impl StorageTrait for BlockStorage {
    fn allocate(&mut self, num_blocks: usize) -> Result<Vec<BlockPointer>, BlockError> {
        self.allocate(num_blocks)
    }
    fn deallocate(&mut self, block_ptrs: &[BlockPointer]) {
        self.deallocate(block_ptrs)
    }
    fn num_free_blocks(&self) -> usize {
        self.num_free_blocks()
    }
}

// ------------------------------------------------------------
// Implementation of the generic BlockManager
// ------------------------------------------------------------
impl<B, S, F> BlockManager<B, S, F>
where
    B: Block,
    S: StorageTrait,
    F: Fn(BlockPointer) -> B,
{
    pub fn new(storage: S, storage_secondary: Option<S>, create_block_fn: F) -> Self {
        Self {
            blocks: HashMap::new(),
            addr_space: HashMap::new(),
            storage,
            storage_secondary,
            addr_assignment: 0,
            create_block_fn,
        }
    }

    pub fn num_free_blocks(&self) -> usize {
        self.storage.num_free_blocks()
    }

    // --- Address space management ---
    pub fn create_address_space(&mut self, inst_id: InstanceId) {
        self.addr_space.insert(inst_id, AddressMap::new());
    }

    pub fn destroy_address_space(&mut self, inst_id: &InstanceId) {
        if let Some(addr_map) = self.addr_space.get(inst_id) {
            let to_remove: Vec<_> = addr_map.mapping.values().copied().collect();
            // For each physical address in that map, deallocate
            for g_addr in to_remove {
                self.deallocate_block(inst_id, g_addr);
            }
        }
        self.addr_space.remove(inst_id);
    }

    // --- Helper for retrieving blocks ---
    pub fn get_block(&self, inst_id: &InstanceId, v_addr: Address) -> Option<&B> {
        self.get_blocks(inst_id, &[v_addr]).get(0).map(|b| *b)
    }

    pub fn get_block_mut(&mut self, inst_id: &InstanceId, v_addr: Address) -> Option<&mut B> {
        // Could factor out the logic or do repeated lookups
        let blocks = self.get_blocks_mut(inst_id, &[v_addr]);
        if !blocks.is_empty() {
            Some(blocks.into_iter().next().unwrap())
        } else {
            None
        }
    }

    pub fn allocate_block(&mut self, inst_id: &InstanceId) -> Result<Address, BlockError> {
        let addrs = self.allocate_blocks(inst_id, 1)?;
        Ok(addrs[0])
    }

    pub fn deallocate_block(&mut self, inst_id: &InstanceId, v_addr: Address) {
        self.delete_blocks(inst_id, &[v_addr]);
    }

    // --- The core methods that do the real work ---
    pub fn get_blocks(&self, inst_id: &InstanceId, v_addrs: &[Address]) -> Vec<&B> {
        let addr_space = match self.addr_space.get(inst_id) {
            Some(space) => space,
            None => return vec![], // or panic/return an error
        };

        let mut blocks = Vec::new();
        for &v_addr in v_addrs {
            let g_addr = addr_space.resolve(v_addr);
            if let Some(block) = self.blocks.get(&g_addr) {
                blocks.push(block);
            }
        }
        blocks
    }

    pub fn get_blocks_mut(&mut self, inst_id: &InstanceId, v_addrs: &[Address]) -> Vec<&mut B> {
        let addr_space = match self.addr_space.get(inst_id) {
            Some(space) => space,
            None => return vec![],
        };

        let mut blocks = Vec::new();
        for &v_addr in v_addrs {
            let g_addr = addr_space.resolve(v_addr);
            if let Some(block) = self.blocks.get_mut(&g_addr) {
                blocks.push(block);
            }
        }
        blocks
    }

    pub fn allocate_blocks(
        &mut self,
        inst_id: &InstanceId,
        num_blocks: usize,
    ) -> Result<Vec<Address>, BlockError> {
        let addr_map = self
            .addr_space
            .get_mut(inst_id)
            .expect("Address space not found for inst_id");

        // 1) Allocate from primary storage
        let ptrs = self.storage.allocate(num_blocks)?;

        // 2) Create block objects
        let mut v_addrs = Vec::new();
        for ptr in ptrs {
            // call the user-supplied function
            let mut block = (self.create_block_fn)(ptr);
            // increment reference count
            block.base_mut().reference_count += 1;

            let g_addr = self.addr_assignment;
            self.addr_assignment += 1;

            self.blocks.insert(g_addr, block);
            let v_addr = addr_map.register(g_addr);
            v_addrs.push(v_addr);
        }

        Ok(v_addrs)
    }

    pub fn allocate_linked_blocks(
        &mut self,
        inst_id: &InstanceId,
        src_inst_id: &InstanceId,
        src_v_addrs: &[Address],
    ) -> Vec<Address> {
        let dst_map = self
            .addr_space
            .get_mut(inst_id)
            .expect("Address space not found for inst_id");
        let src_map = self
            .addr_space
            .get_mut(src_inst_id)
            .expect("Address space not found for src_inst_id");

        let mut dst_v_addrs = Vec::new();
        for &src_v_addr in src_v_addrs {
            let g_addr = src_map.resolve(src_v_addr);
            if let Some(block) = self.blocks.get_mut(&g_addr) {
                block.base_mut().reference_count += 1;
            }

            let new_v_addr = dst_map.register(g_addr);
            dst_v_addrs.push(new_v_addr);
        }

        dst_v_addrs
    }

    pub fn delete_blocks(&mut self, inst_id: &InstanceId, v_addrs: &[Address]) {
        let addr_map = match self.addr_space.get_mut(inst_id) {
            Some(map) => map,
            None => return,
        };

        for &v_addr in v_addrs {
            let g_addr = addr_map.resolve(v_addr);

            if let Some(block) = self.blocks.get_mut(&g_addr) {
                block.base_mut().reference_count -= 1;
            }

            addr_map.unregister(v_addr);

            // If reference_count <= 0, fully remove from manager
            let remove_completely = {
                if let Some(b) = self.blocks.get(&g_addr) {
                    b.base().reference_count <= 0
                } else {
                    false
                }
            };

            if remove_completely {
                if let Some(block) = self.blocks.remove(&g_addr) {
                    // free from primary
                    let ptr = block.base().ptr;
                    self.storage.deallocate(&[ptr]);

                    // if we have a secondary pointer, free it
                    if let Some(ptr2) = block.base().ptr_secondary {
                        if let Some(secondary) = self.storage_secondary.as_mut() {
                            secondary.deallocate(&[ptr2]);
                        }
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------
// KvBlock
// Extends a "block base" with extra fields
// ------------------------------------------------------------
#[derive(Debug)]
pub struct KvBlock {
    base: BlockBase,
    pub position_ids: Vec<usize>,
    pub occupancy: Vec<bool>,
    pub filled: bool,
}

impl KvBlock {
    pub fn new(ptr: BlockPointer) -> Self {
        // This is your old: super().__init__(ptr)
        // plus additional fields
        Self {
            base: BlockBase::new(ptr, None),
            position_ids: vec![],
            occupancy: vec![],
            filled: false,
        }
    }
}

// Implement the Block trait for KvBlock
impl Block for KvBlock {
    fn base(&self) -> &BlockBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BlockBase {
        &mut self.base
    }
}

// A specialized KvBlockStorage that might hold GPU memory in real code
pub struct KvBlockStorage {
    pub base: BlockStorage,
    // Example metadata
    pub num_layers: usize,
    pub num_head: usize,
    pub block_size: usize,
    pub block_dim: usize,
    // placeholders for actual GPU/CPU buffer references...
    // e.g. device, dtype, etc.
}

impl BlockStorage for KvBlockStorage {
    fn max_capacity(&self) -> usize {
        self.base.max_capacity()
    }
    fn num_free_blocks(&self) -> usize {
        self.base.num_free_blocks()
    }
}

impl KvBlockStorage {
    pub fn new(
        max_capacity: usize,
        num_layers: usize,
        num_head: usize,
        block_size: usize,
        block_dim: usize,
    ) -> Self {
        Self {
            base: BlockStorage::new(max_capacity),
            num_layers,
            num_head,
            block_size,
            block_dim,
        }
    }

    // Simulate a "copy" method
    pub fn copy(
        &self,
        src: &KvBlockStorage,
        src_ptr: BlockPointer,
        dst_ptr: BlockPointer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) {
        // In real code, you'd do GPU/CPU memory copying
        // We'll just print or do nothing:
        println!(
            "Copying from src_ptr={} to dst_ptr={}, size={}, offsets=({}, {})",
            src_ptr, dst_ptr, size, src_offset, dst_offset
        );
    }

    pub fn num_bytes(&self) -> usize {
        // Placeholder: your real logic
        self.num_layers * self.block_size * self.block_dim * self.num_head
    }
}

// Implement the StorageTrait for KvBlockStorage
impl StorageTrait for KvBlockStorage {
    fn allocate(&mut self, num_blocks: usize) -> Result<Vec<BlockPointer>, BlockError> {
        self.base.allocate(num_blocks)
    }
    fn deallocate(&mut self, block_ptrs: &[BlockPointer]) {
        self.base.deallocate(block_ptrs)
    }
    fn num_free_blocks(&self) -> usize {
        self.base.num_free_blocks()
    }
}

// A specialized manager for KvBlock
pub struct KvBlockManager {
    // We embed the generic manager
    manager: BlockManager<KvBlock, KvBlockStorage, fn(BlockPointer) -> KvBlock>,
}

impl KvBlockManager {
    pub fn new(storage: KvBlockStorage, storage_secondary: Option<KvBlockStorage>) -> Self {
        fn create_kv_block(ptr: BlockPointer) -> KvBlock {
            let mut b = KvBlock::new(ptr);
            // fill out position_ids, occupancy with default:
            b.position_ids = vec![0; b.base.ptr_secondary.unwrap_or(0)]; // or do something else
            // Actually, we probably want block_size from the storage, but it's not in signature.
            // For demonstration, let's do a simpler approach:
            b.position_ids = vec![0; 0];
            b.occupancy = vec![false; 0];
            b
        }

        let manager = BlockManager::new(storage, storage_secondary, create_kv_block);
        Self { manager }
    }

    // Provide wrappers to call the manager's methods:

    pub fn create_address_space(&mut self, inst_id: InstanceId) {
        self.manager.create_address_space(inst_id);
    }

    pub fn destroy_address_space(&mut self, inst_id: &InstanceId) {
        self.manager.destroy_address_space(inst_id);
    }

    pub fn allocate_block(&mut self, inst_id: &InstanceId) -> Result<Address, BlockError> {
        self.manager.allocate_block(inst_id)
    }

    pub fn deallocate_block(&mut self, inst_id: &InstanceId, v_addr: Address) {
        self.manager.deallocate_block(inst_id, v_addr);
    }

    pub fn copy_tokens(
        &self,
        inst_id: &InstanceId,
        src: Address,
        dst: Address,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) {
        // get the underlying blocks
        if let (Some(src_block), Some(dst_block)) = (
            self.manager.get_block(inst_id, src),
            self.manager.get_block(inst_id, dst),
        ) {
            let src_ptr = src_block.base().ptr;
            let dst_ptr = dst_block.base().ptr;
            // call the storage's copy
            self.manager.storage.copy(
                &self.manager.storage,
                src_ptr,
                dst_ptr,
                src_offset,
                dst_offset,
                size,
            );
        }
    }

    pub fn drop_tokens(
        &mut self,
        inst_id: &InstanceId,
        block_addr: Address,
        start: usize,
        end: usize,
    ) {
        if let Some(block) = self.manager.get_block_mut(inst_id, block_addr) {
            if end <= block.occupancy.len() {
                for occ in block.occupancy[start..end].iter_mut() {
                    *occ = false;
                }
            }
        }
    }
}

// ------------------------------------------------------------
// TokenEmbed block & storage
// ------------------------------------------------------------
#[derive(Debug)]
pub struct TokenEmbed {
    base: BlockBase,
}

impl TokenEmbed {
    pub fn new(ptr: BlockPointer) -> Self {
        Self {
            base: BlockBase::new(ptr, None),
        }
    }
}

impl Block for TokenEmbed {
    fn base(&self) -> &BlockBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BlockBase {
        &mut self.base
    }
}

pub struct TokenEmbedStorage {
    pub base: BlockStorage,
    pub block_dim: usize,
    // placeholders for actual device, dtype, etc...
}

impl TokenEmbedStorage {
    pub fn new(max_capacity: usize, block_dim: usize) -> Self {
        Self {
            base: BlockStorage::new(max_capacity),
            block_dim,
        }
    }

    pub fn num_bytes(&self) -> usize {
        // placeholder: in real code, you'd do something meaningful
        self.block_dim * self.base.max_capacity()
    }
}

impl StorageTrait for TokenEmbedStorage {
    fn allocate(&mut self, num_blocks: usize) -> Result<Vec<BlockPointer>, BlockError> {
        self.base.allocate(num_blocks)
    }
    fn deallocate(&mut self, block_ptrs: &[BlockPointer]) {
        self.base.deallocate(block_ptrs)
    }
    fn num_free_blocks(&self) -> usize {
        self.base.num_free_blocks()
    }
}

// Manager specialized to TokenEmbed
pub struct TokenEmbedManager {
    manager: BlockManager<TokenEmbed, TokenEmbedStorage, fn(BlockPointer) -> TokenEmbed>,
}

impl TokenEmbedManager {
    pub fn new(storage: TokenEmbedStorage, storage_secondary: Option<TokenEmbedStorage>) -> Self {
        fn create_token_embed(ptr: BlockPointer) -> TokenEmbed {
            TokenEmbed::new(ptr)
        }
        let manager = BlockManager::new(storage, storage_secondary, create_token_embed);
        Self { manager }
    }

    // Expose any methods you want
    pub fn allocate_block(&mut self, inst_id: &InstanceId) -> Result<Address, BlockError> {
        self.manager.allocate_block(inst_id)
    }

    pub fn deallocate_block(&mut self, inst_id: &InstanceId, v_addr: Address) {
        self.manager.deallocate_block(inst_id, v_addr);
    }

    // etc... or forward calls
    pub fn create_address_space(&mut self, inst_id: InstanceId) {
        self.manager.create_address_space(inst_id);
    }

    pub fn destroy_address_space(&mut self, inst_id: &InstanceId) {
        self.manager.destroy_address_space(inst_id);
    }
}

// ------------------------------------------------------------
// Example usage
// ------------------------------------------------
fn main() {
    // Example: create a KvBlockManager
    let kv_storage = KvBlockStorage::new(100, 12, 8, 128, 64);
    let mut kv_mgr = KvBlockManager::new(kv_storage, None);

    // Create an address space
    let instance_id = b"my_instance".to_vec();
    kv_mgr.create_address_space(instance_id.clone());

    // Allocate a block
    let v_addr = kv_mgr.allocate_block(&instance_id).unwrap();
    println!("Allocated block at virtual address: {}", v_addr);

    // Drop tokens in that block
    kv_mgr.drop_tokens(&instance_id, v_addr, 10, 20);

    // Deallocate
    kv_mgr.deallocate_block(&instance_id, v_addr);
    kv_mgr.destroy_address_space(&instance_id);
}
