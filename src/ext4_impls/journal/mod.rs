//! Journal (jbd2) engine.
//!
//! Task 1.1: load the journal from inode 8 and map journal-file logical blocks
//! to their physical fs blocks.

use crate::prelude::*;
use crate::ext4_defs::*;
use crate::return_errno_with_message;

// Re-export the jbd2 on-disk codec (defined in `ext4_defs::journal`) so it is
// reachable from the crate root alongside the `Journal` engine. Recovery and the
// test harness assemble/parse/verify log blocks with these.
pub use crate::ext4_defs::journal::{
    assemble_descriptor_block, finalize_commit_csum, finalize_revoke_csum,
    jbd2_block_csum, jbd2_csum_seed, jbd2_data_block_csum, parse_descriptor_block,
    verify_commit_csum, verify_revoke_csum, BlockTag, CommitBlock, JournalSuperblock,
    RevokeBlock, TagFormat, JBD2_COMMIT_BLOCK, JBD2_DESCRIPTOR_BLOCK,
    JBD2_FEATURE_INCOMPAT_64BIT, JBD2_FEATURE_INCOMPAT_CSUM_V2,
    JBD2_FEATURE_INCOMPAT_CSUM_V3, JBD2_FEATURE_INCOMPAT_REVOKE, JBD2_FLAG_DELETED,
    JBD2_FLAG_ESCAPE, JBD2_FLAG_LAST_TAG, JBD2_FLAG_SAME_UUID, JBD2_MAGIC_NUMBER,
    JBD2_REVOKE_BLOCK,
};

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

    /// Read one journal-log block by its log-file block index.
    pub fn read_log_block(&self, fs: &Ext4, idx: Ext4Lblk) -> Result<Vec<u8>> {
        let pblock = self.map_log_block(fs, idx)?;
        let bs = fs.block_size();
        Ok(fs.block_device.read_offset(pblock as usize * bs, bs))
    }

    /// Write one journal-log block by its log-file block index.
    pub fn write_log_block(&self, fs: &Ext4, idx: Ext4Lblk, data: &[u8]) -> Result<()> {
        let pblock = self.map_log_block(fs, idx)?;
        let bs = fs.block_size();
        // data must be exactly one block; guard rather than silently truncate/pad.
        if data.len() != bs {
            return_errno_with_message!(Errno::EINVAL, "write_log_block: data not one block");
        }
        fs.block_device.write_offset(pblock as usize * bs, data);
        Ok(())
    }
}
