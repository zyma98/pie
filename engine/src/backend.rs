use crate::remote_obj::{
    Addr, BlockError, IdPool, KvBlock, KvBlockAllocator, ObjectAllocator, RemoteObjId, TensorKind,
    TokenEmb,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use uuid::Uuid;

type StreamId = Uuid;
#[derive(Debug)]
enum Command {


    // Embs
    AllocateEmb(usize),
    DeallocateEmb(usize),

    // KvBlocks
    AllocateKvBlock(usize),
    DeallocateKvBlock(usize),
    CopyKvBlock(usize, usize, usize, usize, usize),
    MaskKvBlock(usize, Vec<bool>),

    Allocate(Addr),
    Deallocate(Addr),
    Copy(Addr, Addr, usize, usize, usize),
}

#[derive(Debug, Clone)]
pub struct Backend {
    block_size: usize,
    inner: Rc<RefCell<BackendInner>>,
}

#[derive(Debug)]
struct BackendInner {
    block_size: usize,
    kv_block_id_pool: IdPool<RemoteObjId>,
    emb_id_pool: IdPool<RemoteObjId>,
    cmd_queue: Vec<(StreamId, Command)>,
}

impl Backend {
    pub fn new(block_size: usize, max_kv_blocks: usize, max_embs: usize) -> Self {
        Self {
            block_size,
            inner: Rc::new(RefCell::new(BackendInner {
                block_size,
                kv_block_id_pool: IdPool::new(max_kv_blocks),
                emb_id_pool: IdPool::new(max_embs),
                cmd_queue: Vec::new(),
            })),
        }
    }

    pub fn enqueue_cmd(&self, stream_id: StreamId, cmd: Command) -> Result<(), BlockError> {
        self.inner.borrow_mut().cmd_queue.push((stream_id, cmd));
        Ok(())
    }
}

impl ObjectAllocator<TokenEmb> for Backend {
    fn alloc(&self, stream_id: StreamId) -> Result<RemoteObjId, BlockError> {
        let new_obj_id = self
            .inner
            .borrow_mut()
            .emb_id_pool
            .acquire()
            .ok_or(BlockError::NoFreeBlocks)?;

        let cmd = Command::AllocateEmb(new_obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.inner.borrow_mut().emb_id_pool.release(obj_id)?;

        let cmd = Command::DeallocateEmb(obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(())
    }

    fn available(&self) -> usize {
        self.inner.borrow().emb_id_pool.available()
    }
}

impl ObjectAllocator<KvBlock> for Backend {
    fn alloc(&self, stream_id: StreamId) -> Result<RemoteObjId, BlockError> {
        let new_obj_id = self
            .inner
            .borrow_mut()
            .kv_block_id_pool
            .acquire()
            .ok_or(BlockError::NoFreeBlocks)?;

        let cmd = Command::AllocateKvBlock(new_obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(new_obj_id)
    }

    fn dealloc(&self, stream_id: StreamId, obj_id: RemoteObjId) -> Result<(), BlockError> {
        // Release the object id back to the pool.
        self.inner.borrow_mut().kv_block_id_pool.release(obj_id)?;

        let cmd = Command::DeallocateKvBlock(obj_id);
        self.enqueue_cmd(stream_id, cmd)?;

        Ok(())
    }

    fn available(&self) -> usize {
        self.inner.borrow().kv_block_id_pool.available()
    }
}

impl KvBlockAllocator for Backend {
    fn copy_tokens(
        &self,
        stream_id: StreamId,
        src_ptr: RemoteObjId,
        dst_ptr: RemoteObjId,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<(), BlockError> {
        let cmd = Command::CopyKvBlock(src_ptr, dst_ptr, src_offset, dst_offset, size);
        
        self.enqueue_cmd(stream_id, cmd)?;
        Ok(())
    }

    fn mask_tokens(&self, ptr: RemoteObjId, mask: &[bool]) -> Result<(), BlockError> {
        todo!()
    }
}
