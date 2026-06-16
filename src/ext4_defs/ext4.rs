use crate::prelude::*;
use crate::ext4_impls::journal::{Journal, JournalDevice};

use super::*;

#[derive(Debug, Clone)]
pub struct SystemZone {
    pub group: u32,
    pub start_blk: u64,
    pub end_blk: u64,
}

/// POSIX `fcntl` lock types, matching the kernel `F_RDLCK`/`F_WRLCK`/`F_UNLCK`
/// values FUSE passes to `getlk`/`setlk`.
pub const F_RDLCK: i32 = 0;
pub const F_WRLCK: i32 = 1;
pub const F_UNLCK: i32 = 2;

/// One advisory byte-range lock. `end` is inclusive (`u64::MAX` for "to EOF",
/// the convention FUSE uses for `l_len == 0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FileLock {
    pub owner: u64,
    pub pid: u32,
    pub start: u64,
    pub end: u64,
    pub typ: i32,
}

pub struct Ext4 {
    pub block_device: Arc<dyn BlockDevice>,
    pub super_block: Ext4Superblock,
    pub system_zone_cache: Option<Vec<SystemZone>>,
    /// In-memory advisory byte-range locks, keyed by inode number. Like the
    /// kernel's, these are process/runtime state and are never persisted to
    /// disk. Maintained by `fuse_setlk`/`fuse_getlk`.
    pub locks: BTreeMap<u32, Vec<FileLock>>,
    /// Typed handle to the journaling block-device wrapper (Some when opened via
    /// `open_journaled`). Used to begin/end the running transaction.
    pub journal_device: Option<Arc<JournalDevice>>,
    /// The journal engine (Some when a journal is present and journaling is on).
    pub journal: Option<Journal>,
}
