//! Journal (jbd2) engine.
//!
//! Task 1.1: load the journal from inode 8 and map journal-file logical blocks
//! to their physical fs blocks.

use crate::prelude::*;
use crate::ext4_defs::*;

/// The in-memory journal engine, anchored on the on-disk jbd2 superblock and
/// the inode that backs the journal file (inode 8).
pub struct Journal {
    /// The parsed jbd2 superblock (logical block 0 of the journal inode).
    pub sb: JournalSuperblock,
    /// Inode ref for journal inode 8, used to map log blocks to physical
    /// blocks. `Ext4InodeRef` is `Clone`, so we store it directly rather than
    /// re-fetching by inode number on every map.
    pub inode_ref: Ext4InodeRef,
}

impl Journal {
    /// Load the journal from journal inode 8. Returns:
    /// - `Ok(None)` if the fs has no `has_journal` feature, or the journal
    ///   superblock has a bad magic (treated as "no journal", per design).
    /// - `Ok(Some(journal))` if a valid journal superblock was parsed.
    /// - `Err(_)` on an I/O error reading the inode/block.
    pub fn load(fs: &Ext4) -> Result<Option<Self>> {
        if !fs.super_block.has_feature_journal() {
            return Ok(None);
        }
        let inode_ref = fs.get_inode_ref(JOURNAL_INODE);
        // journal superblock = logical block 0 of inode 8
        let pblock = fs.get_pblock_idx(&inode_ref, 0)?;
        let bs = fs.block_size();
        let buf = fs.block_device.read_offset(pblock as usize * bs, bs);
        match JournalSuperblock::parse(&buf) {
            Ok(sb) => Ok(Some(Journal { sb, inode_ref })),
            Err(_) => Ok(None), // bad magic -> treat as no journal
        }
    }

    /// Map a journal-file logical block index to its physical fs block.
    pub fn map_log_block(&self, fs: &Ext4, log_block: Ext4Lblk) -> Result<Ext4Fsblk> {
        fs.get_pblock_idx(&self.inode_ref, log_block)
    }
}
