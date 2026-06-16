//! Write-capturing `BlockDevice` wrapper (jbd2 write side).
//!
//! Task 3.2: `JournalDevice` wraps the inner block device and, while a
//! transaction is open, captures every `write_offset` into the running
//! `Transaction` as whole (read-modify-written) blocks instead of letting it hit
//! the inner device. Reads overlay staged blocks for read-your-writes
//! consistency. With no transaction open it is a transparent pass-through.

use crate::prelude::*;
use crate::ext4_defs::BlockDevice;
use super::transaction::Transaction;
use spin::Mutex;

/// Interior, mutex-guarded state: the running transaction (if any) and the
/// begin/end nesting depth.
struct DevState {
    txn: Option<Transaction>,
    depth: u32,
}

/// A `BlockDevice` wrapper that captures writes into a running journal
/// transaction. While a transaction is open (begin without matching end),
/// `write_offset` stages whole (read-modify-written) blocks into the transaction
/// instead of hitting the inner device, and `read_offset` overlays staged blocks
/// for read-your-writes consistency. With no transaction open it is a
/// transparent pass-through.
pub struct JournalDevice {
    inner: Arc<dyn BlockDevice>,
    block_size: usize,
    state: Mutex<DevState>,
}

impl JournalDevice {
    pub fn new(inner: Arc<dyn BlockDevice>, block_size: usize) -> Self {
        JournalDevice {
            inner,
            block_size,
            state: Mutex::new(DevState { txn: None, depth: 0 }),
        }
    }

    /// The inner device (used by commit/checkpoint in Phase 4 to write the log
    /// and final locations directly, bypassing capture).
    pub fn inner(&self) -> &Arc<dyn BlockDevice> {
        &self.inner
    }

    /// Begin (or nest into) a transaction at `sequence`. Nested begins share the
    /// transaction created by the outermost begin; `sequence` is used only when
    /// creating it.
    pub fn begin(&self, sequence: u32) {
        let mut s = self.state.lock();
        if s.depth == 0 {
            s.txn = Some(Transaction::new(sequence));
        }
        s.depth += 1;
    }

    /// End one begin level. When the outermost level closes, returns the running
    /// transaction (to be committed by the caller); otherwise None (still
    /// nested).
    pub fn end(&self) -> Option<Transaction> {
        let mut s = self.state.lock();
        if s.depth == 0 {
            return None;
        }
        s.depth -= 1;
        if s.depth == 0 {
            s.txn.take()
        } else {
            None
        }
    }

    pub fn is_active(&self) -> bool {
        self.state.lock().depth > 0
    }

    /// Record a revoke in the running transaction (no-op if none open).
    pub fn revoke(&self, block: u64) {
        let mut s = self.state.lock();
        if let Some(t) = s.txn.as_mut() {
            t.revoke(block);
        }
    }
}

impl BlockDevice for JournalDevice {
    fn read_offset(&self, offset: usize, len: usize) -> Vec<u8> {
        let s = self.state.lock();
        if let Some(t) = s.txn.as_ref() {
            // Base content from the inner device, then overlay any staged blocks
            // the range touches (read-your-writes).
            let mut result = self.inner.read_offset(offset, len);
            let bs = self.block_size;
            let mut pos = offset;
            let mut dst = 0usize;
            while dst < len {
                let b = (pos / bs) as u64;
                let intra = pos % bs;
                let n = core::cmp::min(bs - intra, len - dst);
                if let Some(sb) = t.blocks.get(&b) {
                    result[dst..dst + n].copy_from_slice(&sb.data[intra..intra + n]);
                }
                pos += n;
                dst += n;
            }
            result
        } else {
            self.inner.read_offset(offset, len)
        }
    }

    fn write_offset(&self, offset: usize, data: &[u8]) {
        let mut s = self.state.lock();
        if s.txn.is_some() {
            let bs = self.block_size;
            let mut pos = offset;
            let mut src = 0usize;
            while src < data.len() {
                let b = (pos / bs) as u64;
                let intra = pos % bs;
                let n = core::cmp::min(bs - intra, data.len() - src);

                // Full current block: prefer the staged copy if present, else read
                // the whole block from the inner device. Compute the merged block
                // into a local first (immutable borrow of `txn` scoped here), then
                // stage it via a separate mutable borrow — keeps the borrow checker
                // happy without changing the logic.
                let mut block = match s.txn.as_ref().unwrap().blocks.get(&b) {
                    Some(sb) => sb.data.clone(),
                    None => self.inner.read_offset(b as usize * bs, bs),
                };
                block[intra..intra + n].copy_from_slice(&data[src..src + n]);

                // Default tag = data (false); metadata tagging is Phase 5.
                s.txn.as_mut().unwrap().stage(b, block, false);

                pos += n;
                src += n;
            }
        } else {
            self.inner.write_offset(offset, data);
        }
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Vec-backed `BlockDevice` for tests (interior mutability via spin::Mutex).
    struct MemDev {
        data: Mutex<Vec<u8>>,
    }
    impl MemDev {
        fn new(size: usize) -> Self {
            MemDev {
                data: Mutex::new(vec![0u8; size]),
            }
        }
    }
    impl BlockDevice for MemDev {
        fn read_offset(&self, off: usize, len: usize) -> Vec<u8> {
            let d = self.data.lock();
            d[off..off + len].to_vec()
        }
        fn write_offset(&self, off: usize, data: &[u8]) {
            let mut d = self.data.lock();
            d[off..off + data.len()].copy_from_slice(data);
        }
    }

    // `JournalDevice` must be `Send + Sync` to be usable as `Arc<dyn BlockDevice>`
    // (the trait requires it). This is a compile-time assertion: spin::Mutex<T> is
    // Sync where T: Send, and DevState/Transaction are Send.
    fn _assert_send_sync<T: Send + Sync>() {}
    const _: () = {
        let _ = _assert_send_sync::<JournalDevice>;
    };

    #[test]
    fn passthrough_when_no_transaction() {
        let inner = Arc::new(MemDev::new(4096));
        let jd = JournalDevice::new(inner.clone(), 512);
        jd.write_offset(512, &vec![0xAB; 512]); // block 1
        assert_eq!(inner.read_offset(512, 512), vec![0xAB; 512]); // hit the inner device
        assert_eq!(jd.read_offset(512, 512), vec![0xAB; 512]);
    }

    #[test]
    fn captures_whole_block_write_without_touching_inner() {
        let inner = Arc::new(MemDev::new(4096));
        let jd = JournalDevice::new(inner.clone(), 512);
        jd.begin(1);
        jd.write_offset(512, &vec![0xCD; 512]); // block 1
        // inner device NOT modified (write was captured):
        assert_eq!(inner.read_offset(512, 512), vec![0x00; 512]);
        // read-your-writes through the wrapper sees the staged data:
        assert_eq!(jd.read_offset(512, 512), vec![0xCD; 512]);
        let txn = jd.end().expect("outermost end returns txn");
        assert_eq!(txn.blocks.len(), 1);
        assert_eq!(txn.blocks[&1].data, vec![0xCD; 512]);
    }

    #[test]
    fn sub_block_write_read_modify_writes_full_block() {
        let inner = Arc::new(MemDev::new(4096));
        // seed block 1 on the inner device with a known background.
        inner.write_offset(512, &vec![0x11; 512]);
        let jd = JournalDevice::new(inner.clone(), 512);
        jd.begin(1);
        // write 4 bytes at intra-block offset 100 of block 1 (device offset 612).
        jd.write_offset(612, &[0xFF, 0xFF, 0xFF, 0xFF]);
        let txn = jd.end().unwrap();
        let staged = &txn.blocks[&1].data;
        assert_eq!(staged.len(), 512);
        assert_eq!(&staged[100..104], &[0xFF; 4]); // the 4 changed bytes
        assert_eq!(staged[0], 0x11); // background preserved
        assert_eq!(staged[104], 0x11);
        // inner device still untouched by the captured write.
        assert_eq!(inner.read_offset(512, 512), vec![0x11; 512]);
    }

    #[test]
    fn nested_begin_end_shares_one_transaction() {
        let inner = Arc::new(MemDev::new(4096));
        let jd = JournalDevice::new(inner.clone(), 512);
        jd.begin(1);
        jd.begin(1); // nested
        jd.write_offset(0, &vec![0x22; 512]);
        assert!(jd.end().is_none()); // inner level: no txn yet
        let txn = jd.end().expect("outer level returns the shared txn");
        assert_eq!(txn.blocks.len(), 1);
    }
}
