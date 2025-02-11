use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;
use std::sync::atomic::{AtomicIsize, Ordering};

use num_traits::PrimInt;

use uuid::Uuid;

pub type InstanceId = Uuid;
pub type Addr = usize;

// ------------------------------------------------------------

pub type RemoteObjId = usize;

// this helps the backend to make optimization decisions.
pub enum TensorKind {
    KvBlock,
    TokenEmb,
    Dist,
    Flex,
}

pub trait RemoteObj {
    fn new(id: RemoteObjId) -> Self;

    fn kind() -> TensorKind;

    fn id(&self) -> RemoteObjId;

    // reference counter
    fn rc(&self) -> &Counter;
}

pub trait ObjectAllocator<T: RemoteObj> {
    fn alloc(&self, stream_id: &InstanceId) -> Result<RemoteObjId, BlockError>;

    fn dealloc(&self, stream_id: &InstanceId, obj_id: RemoteObjId) -> Result<(), BlockError>;

    fn available(&self) -> usize;
}

// reference-counted object manager
pub trait RcObjectManager<T: RemoteObj, B: ObjectAllocator<T>> {
    fn objects(&self) -> &HashMap<RemoteObjId, T>;

    fn objects_mut(&mut self) -> &mut HashMap<RemoteObjId, T>;

    fn addr_maps(&self) -> &HashMap<InstanceId, AddrMap<Addr, RemoteObjId>>;

    fn addr_maps_mut(&mut self) -> &mut HashMap<InstanceId, AddrMap<Addr, RemoteObjId>>;

    fn backend(&self) -> &B;

    fn init_instance(&mut self, inst_id: InstanceId) -> Result<(), BlockError> {
        if self.addr_maps().contains_key(&inst_id) {
            return Err(BlockError::InstanceAlreadyExists);
        }

        self.addr_maps_mut().insert(inst_id, AddrMap::new());
        Ok(())
    }

    fn destroy_instance(&mut self, inst_id: &InstanceId) -> Result<(), BlockError> {
        // Collect the keys into a Vec, which drops the borrow on `self` after collection.
        let addrs: Vec<_> = self
            .addr_maps()
            .get(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .mapping
            .keys()
            .copied() // Copy the keys to get owned values (assuming keys are Copy)
            .collect();

        // Now iterate over the owned keys, and it's safe to call mutable methods on `self`
        for addr in addrs {
            self.dealloc(inst_id, addr)?;
        }

        self.addr_maps_mut().remove(&inst_id);

        Ok(())
    }

    fn alloc(&mut self, inst_id: &InstanceId) -> Result<Addr, BlockError> {
        let new_g_addr = self.backend().alloc(inst_id)?;

        let new_addr = self
            .addr_maps_mut()
            .get_mut(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .register(new_g_addr);

        let new_obj = T::new(new_g_addr);
        new_obj.rc().inc();

        self.objects_mut().insert(new_g_addr, new_obj);
        Ok(new_addr)
    }

    fn create_ref(
        &mut self,
        inst_id: &InstanceId,
        src_inst_id: &InstanceId,
        src_addr: Addr,
    ) -> Result<Addr, BlockError> {
        let src_g_addr = self
            .addr_maps()
            .get(&src_inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .resolve(src_addr)?;

        // insert a new virtual address
        let new_v_addr = self
            .addr_maps_mut()
            .get_mut(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .register(src_g_addr);

        // increase ref count
        self.objects()
            .get(&src_g_addr)
            .ok_or(BlockError::BlockNotFound)?
            .rc()
            .inc();

        Ok(new_v_addr)
    }

    fn dealloc(&mut self, inst_id: &InstanceId, addr: Addr) -> Result<(), BlockError> {
        // remove and get the global address
        let g_addr = self
            .addr_maps_mut()
            .get_mut(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .unregister(addr)?;

        let remove_entirely = self
            .objects()
            .get(&g_addr)
            .ok_or(BlockError::BlockNotFound)?
            .rc()
            .dec()
            <= 0;

        // remove the block if the ref count is 0
        if remove_entirely {
            self.backend().dealloc(inst_id, g_addr)?;
            self.objects_mut().remove(&g_addr);
        }

        Ok(())
    }

    fn get(&self, inst_id: &InstanceId, addr: Addr) -> Result<&T, BlockError> {
        let g_addr = self
            .addr_maps()
            .get(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .resolve(addr)?;
        self.objects().get(&g_addr).ok_or(BlockError::BlockNotFound)
    }

    fn get_mut(&mut self, inst_id: &InstanceId, addr: Addr) -> Result<&mut T, BlockError> {
        let g_addr = self
            .addr_maps()
            .get(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .resolve(addr)?;
        self.objects_mut()
            .get_mut(&g_addr)
            .ok_or(BlockError::BlockNotFound)
    }

    fn get_many(&self, inst_id: &InstanceId, addrs: &[Addr]) -> Result<Vec<&T>, BlockError> {
        let mut objs = Vec::with_capacity(addrs.len());
        for addr in addrs {
            let g_addr = self
                .addr_maps()
                .get(&inst_id)
                .ok_or(BlockError::InstanceNotFound)?
                .resolve(*addr)?;
            objs.push(
                self.objects()
                    .get(&g_addr)
                    .ok_or(BlockError::BlockNotFound)?,
            );
        }
        Ok(objs)
    }

    fn available_objs(&self) -> usize {
        self.backend().available()
    }
}

// ------------------------------------------------------------

pub struct KvBlock {
    id: RemoteObjId,
    references: Counter,

    // pos ids and vacancy maps
    pub position_ids: Vec<u32>,
    pub occupied: Vec<bool>,
    pub filled: bool,
}

pub trait KvBlockAllocator: ObjectAllocator<KvBlock> {
    fn copy_tokens(
        &self,
        stream_id: &InstanceId,
        src_ptr: RemoteObjId,
        dst_ptr: RemoteObjId,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<(), BlockError>;

    fn mask_tokens(&self, ptr: RemoteObjId, mask: &[bool]) -> Result<(), BlockError>;
}

impl RemoteObj for KvBlock {
    fn new(id: RemoteObjId) -> Self {
        KvBlock {
            id,
            references: Counter::new(0),
            position_ids: Vec::new(),
            occupied: Vec::new(),
            filled: false,
        }
    }

    fn kind() -> TensorKind {
        TensorKind::KvBlock
    }

    fn id(&self) -> RemoteObjId {
        self.id
    }

    fn rc(&self) -> &Counter {
        &self.references
    }
}

pub struct KvBlockManager<B> {
    kv_blocks: HashMap<RemoteObjId, KvBlock>,
    token_embs: HashMap<RemoteObjId, TokenEmb>,
    virtual_addr_maps: HashMap<InstanceId, AddrMap<Addr, RemoteObjId>>,

    // primary storage
    backend: B,
}

impl<B> KvBlockManager<B>
where
    B: KvBlockAllocator,
{
    pub fn new(backend: B) -> Self {
        Self {
            kv_blocks: HashMap::new(),
            token_embs: HashMap::new(),
            virtual_addr_maps: HashMap::new(),
            backend,
        }
    }

    pub fn copy_tokens(
        &mut self,
        inst_id: &InstanceId,
        src_addr: Addr,
        dst_addr: Addr,
        src_token_offset: usize,
        dst_token_offset: usize,
        size: usize,
    ) -> Result<(), BlockError> {
        let src_block_ptr = self.get(inst_id, src_addr)?.id();
        let dst_block_ptr = self.get(inst_id, dst_addr)?.id();

        self.backend().copy_tokens(
            inst_id,
            src_block_ptr,
            dst_block_ptr,
            src_token_offset,
            dst_token_offset,
            size,
        )?;

        // First, get a temporary copy of the data from the source block.
        // Here we use an immutable borrow so that we can later borrow the destination mutably.
        let (src_position_ids, src_occupied) = {
            let src_block = self.get(inst_id, src_addr)?;
            (
                src_block.position_ids[src_token_offset..src_token_offset + size].to_vec(),
                src_block.occupied[src_token_offset..src_token_offset + size].to_vec(),
            )
        };

        // Now get a mutable borrow of the destination block and update its data.
        let dst_block = self.get_mut(inst_id, dst_addr)?;
        for i in 0..size {
            dst_block.position_ids[dst_token_offset + i] = src_position_ids[i];
            dst_block.occupied[dst_token_offset + i] = src_occupied[i];
        }
        Ok(())
    }

    pub fn mask_tokens(
        &mut self,
        inst_id: &InstanceId,
        virtual_addr: Addr,
        mask: &[bool],
    ) -> Result<(), BlockError> {
        let block_id = {
            let block = self.get_mut(inst_id, virtual_addr)?;
            for i in 0..mask.len() {
                block.occupied[i] = mask[i];
            }
            block.id()
        };
        self.backend.mask_tokens(block_id, mask)?;

        Ok(())
    }
}

impl<B> RcObjectManager<KvBlock, B> for KvBlockManager<B>
where
    B: ObjectAllocator<KvBlock>,
{
    fn objects(&self) -> &HashMap<Addr, KvBlock> {
        &self.kv_blocks
    }

    fn objects_mut(&mut self) -> &mut HashMap<Addr, KvBlock> {
        &mut self.kv_blocks
    }

    fn addr_maps(&self) -> &HashMap<InstanceId, AddrMap<Addr, RemoteObjId>> {
        &self.virtual_addr_maps
    }

    fn addr_maps_mut(&mut self) -> &mut HashMap<InstanceId, AddrMap<Addr, RemoteObjId>> {
        &mut self.virtual_addr_maps
    }

    fn backend(&self) -> &B {
        &self.backend
    }
}

// ------------------------------------------------------------

pub struct TokenEmb {
    ptr: RemoteObjId,
    reference: Counter,
}

impl RemoteObj for TokenEmb {
    fn new(obj_id: RemoteObjId) -> Self {
        TokenEmb {
            ptr: obj_id,
            reference: Counter::new(0),
        }
    }

    fn kind() -> TensorKind {
        TensorKind::TokenEmb
    }

    fn id(&self) -> RemoteObjId {
        self.ptr
    }

    fn rc(&self) -> &Counter {
        &self.reference
    }
}

pub struct TokenEmbManager<B> {
    token_embs: HashMap<RemoteObjId, TokenEmb>,
    virtual_addr_maps: HashMap<InstanceId, AddrMap<Addr, RemoteObjId>>,

    // primary storage
    storage: B,
}

impl<B> TokenEmbManager<B> {
    pub fn new(storage: B) -> Self {
        Self {
            token_embs: HashMap::new(),
            virtual_addr_maps: HashMap::new(),
            storage,
        }
    }
}

impl<B> RcObjectManager<TokenEmb, B> for TokenEmbManager<B>
where
    B: ObjectAllocator<TokenEmb>,
{
    fn objects(&self) -> &HashMap<Addr, TokenEmb> {
        &self.token_embs
    }

    fn objects_mut(&mut self) -> &mut HashMap<Addr, TokenEmb> {
        &mut self.token_embs
    }

    fn addr_maps(&self) -> &HashMap<InstanceId, AddrMap<Addr, RemoteObjId>> {
        &self.virtual_addr_maps
    }

    fn addr_maps_mut(&mut self) -> &mut HashMap<InstanceId, AddrMap<Addr, RemoteObjId>> {
        &mut self.virtual_addr_maps
    }

    fn backend(&self) -> &B {
        &self.storage
    }
}

// ------------------------------------------------------------

#[derive(Debug)]
pub enum BlockError {
    NoFreeBlocks,
    NotEnoughFreeBlocks { requested: usize, available: usize },
    VirtualAddressTranslationFailed,
    BlockNotFound,
    InstanceNotFound,
    InstanceAlreadyExists,
    ResourceNotFound,
    ResourcePermissionDenied,
}

// ------------------------------------------------------------
// AddressMap
// ------------------------------------------------------------
#[derive(Debug)]
pub struct AddrMap<K, V> {
    mapping: HashMap<K, V>,
    addr_pool: IdPool<K>,
}

impl<K, V> AddrMap<K, V>
where
    K: PrimInt + Hash,
    V: Copy,
{
    pub fn new() -> Self {
        Self {
            mapping: HashMap::new(),
            addr_pool: IdPool::new(K::max_value()),
        }
    }

    pub fn register(&mut self, g_addr: V) -> K {
        let new_v_addr = self.addr_pool.acquire().unwrap();
        self.mapping.insert(new_v_addr, g_addr);
        new_v_addr
    }

    pub fn unregister(&mut self, v_addr: K) -> Result<V, BlockError> {
        let removed_g_addr = self
            .mapping
            .remove(&v_addr)
            .ok_or(BlockError::BlockNotFound)?;
        self.addr_pool.release(v_addr)?;

        Ok(removed_g_addr)
    }

    pub fn resolve(&self, v_addr: K) -> Result<V, BlockError> {
        self.mapping
            .get(&v_addr)
            .copied()
            .ok_or(BlockError::VirtualAddressTranslationFailed)
    }

    pub fn resolve_many(&self, v_addrs: &[K]) -> Result<Vec<V>, BlockError> {
        let mut g_addrs = Vec::with_capacity(v_addrs.len());
        for v_addr in v_addrs {
            let global_addr = self.resolve(v_addr.clone())?;
            g_addrs.push(global_addr);
        }
        Ok(g_addrs)
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
    pub fn inc(&self) -> isize {
        // Using relaxed ordering because we only care about the counter's value.
        self.count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements the counter by 1.
    pub fn dec(&self) -> isize {
        self.count.fetch_sub(1, Ordering::Relaxed) - 1
    }

    /// Returns the current count.
    pub fn get(&self) -> isize {
        self.count.load(Ordering::Relaxed)
    }
}

/// A very fast bounded ID pool that always returns the smallest available ID.
/// The pool is created with a maximum capacity.
#[derive(Debug)]
pub struct IdPool<T> {
    /// The next (never‑allocated) ID.
    next: T,
    /// The set of freed IDs.
    free: BTreeSet<T>,
    /// The maximum number of IDs that can be allocated.
    max_capacity: T,
}

impl<T> IdPool<T>
where
    T: PrimInt,
{
    /// Create a new ID pool with the given maximum capacity.
    pub fn new(max_capacity: T) -> Self {
        Self {
            next: T::zero(),
            free: BTreeSet::new(),
            max_capacity,
        }
    }

    /// Allocate and return the smallest available ID.
    ///
    /// Returns `Some(id)` if an ID is available, or `None` if the pool is exhausted.
    pub fn acquire(&mut self) -> Option<T> {
        if let Some(&id) = self.free.iter().next() {
            // There is a freed ID available. Remove and return it.
            self.free.remove(&id);
            Some(id)
        } else if self.next < self.max_capacity {
            // No freed IDs available; allocate a fresh one.
            let addr = self.next;
            self.next = self.next + T::one();
            Some(addr)
        } else {
            // Pool is exhausted.
            None
        }
    }

    /// Release an ID back into the pool so it can be re-used.

    pub fn release(&mut self, addr: T) -> Result<(), BlockError> {
        // Only allow releasing IDs that were allocated.
        if addr >= self.next {
            return Err(BlockError::VirtualAddressTranslationFailed);
        }

        // Insert the id into the free set.
        self.free.insert(addr);

        // Tail optimization: if the largest freed id is exactly next-1,
        // collapse the free block by decrementing `next`.
        while let Some(&last) = self.free.iter().next_back() {
            if last == self.next - T::one() {
                self.free.remove(&last);
                self.next = self.next - T::one();
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
        self.free.len() + (self.max_capacity - self.next).to_usize().unwrap()
    }
}
