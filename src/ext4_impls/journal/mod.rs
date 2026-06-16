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

/// One block logged by a committed transaction: its final on-disk location and
/// the log block holding its data, plus tag flags (ESCAPE handling at replay).
pub struct LoggedBlock {
    /// The block's final on-disk location (from the descriptor tag).
    pub final_block: u64,
    /// The log block holding this block's data (immediately follows the descriptor).
    pub data_log_block: u64,
    /// jbd2 tag flags (e.g. `JBD2_FLAG_ESCAPE`).
    pub flags: u16,
}

/// A transaction confirmed committed during SCAN.
pub struct ScannedTxn {
    /// The transaction's commit ID (h_sequence).
    pub sequence: u32,
    /// Blocks logged by this transaction, in log order.
    pub blocks: Vec<LoggedBlock>,
    /// Log block indices of revoke blocks belonging to this transaction.
    pub revoke_log_blocks: Vec<u64>,
}

/// Result of the SCAN pass (jbd2 PASS_SCAN): the committed transactions, in log
/// order, and the sequence of the last committed transaction.
pub struct ScanResult {
    /// Committed transactions, in log order.
    pub txns: Vec<ScannedTxn>,
    /// Sequence of the last committed txn. If none were committed this is
    /// `sb.sequence - 1` (one before the first expected commit).
    pub last_sequence: u32,
}

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

    /// SCAN pass (jbd2 PASS_SCAN, adapted): walk the live region of the log from
    /// `sb.start` and collect every fully-committed transaction. A transaction is
    /// a run of descriptor/data/revoke blocks (each carrying the expected
    /// `h_sequence`) terminated by a valid COMMIT block. The log ends at the first
    /// block whose magic mismatches, whose sequence belongs to an older
    /// generation, or at a torn/invalid commit — and any trailing partial
    /// transaction (logged but not committed) is discarded, exactly as jbd2 does.
    pub fn scan(&self, fs: &Ext4) -> Result<ScanResult> {
        // Recovery must reflect the *current* on-disk superblock: s_start and
        // s_sequence are set when the journal goes dirty, after `Journal::load`
        // cached `self.sb`. Re-read log block 0 so the walk anchors on the live
        // (dirty) region rather than the stale cached state.
        let sb_buf = self.read_log_block(fs, 0)?;
        let sb = JournalSuperblock::parse(&sb_buf)?;

        let fmt = TagFormat::from_features(sb.feature_incompat);
        let has_64bit = sb.feature_incompat & JBD2_FEATURE_INCOMPAT_64BIT != 0;
        let has_csum = sb.feature_incompat
            & (JBD2_FEATURE_INCOMPAT_CSUM_V2 | JBD2_FEATURE_INCOMPAT_CSUM_V3)
            != 0;
        let seed = jbd2_csum_seed(&sb.uuid);

        let first = sb.first;
        let maxlen = sb.maxlen;
        let mut cursor = sb.start;
        let mut next_seq = sb.sequence;

        // s_start == 0 means the journal is clean: nothing to recover.
        if sb.start == 0 {
            return Ok(ScanResult {
                txns: Vec::new(),
                last_sequence: next_seq.wrapping_sub(1),
            });
        }

        let mut txns: Vec<ScannedTxn> = Vec::new();
        // Accumulators for the (not-yet-committed) transaction at `next_seq`.
        let mut cur_blocks: Vec<LoggedBlock> = Vec::new();
        let mut cur_revokes: Vec<u64> = Vec::new();

        // Bound the walk by maxlen iterations so a corrupt log can't loop forever.
        for _ in 0..maxlen {
            let block = self.read_log_block(fs, cursor)?;
            // journal_header_t: h_magic@0, h_blocktype@4, h_sequence@8 (all BE u32).
            let magic = u32::from_be_bytes([block[0], block[1], block[2], block[3]]);
            if magic != JBD2_MAGIC_NUMBER {
                break; // end of log
            }
            let blocktype = u32::from_be_bytes([block[4], block[5], block[6], block[7]]);
            let h_seq = u32::from_be_bytes([block[8], block[9], block[10], block[11]]);
            if h_seq != next_seq {
                break; // older/overwritten generation: the live log ends here
            }

            match blocktype {
                JBD2_DESCRIPTOR_BLOCK => {
                    let tags = parse_descriptor_block(&block, fmt, has_64bit)?;
                    // Advance past the descriptor; each tag's data block is the
                    // block immediately following, in order.
                    cursor += 1;
                    if cursor >= maxlen {
                        cursor = first;
                    }
                    for tag in tags {
                        cur_blocks.push(LoggedBlock {
                            final_block: tag.blocknr,
                            data_log_block: cursor as u64,
                            flags: tag.flags,
                        });
                        cursor += 1;
                        if cursor >= maxlen {
                            cursor = first;
                        }
                    }
                    continue; // cursor already advanced+wrapped
                }
                JBD2_REVOKE_BLOCK => {
                    cur_revokes.push(cursor as u64);
                }
                JBD2_COMMIT_BLOCK => {
                    // A torn/invalid commit ends the log without committing the
                    // pending transaction (its accumulated blocks are discarded).
                    if has_csum && !verify_commit_csum(&block, seed) {
                        break;
                    }
                    txns.push(ScannedTxn {
                        sequence: next_seq,
                        blocks: core::mem::take(&mut cur_blocks),
                        revoke_log_blocks: core::mem::take(&mut cur_revokes),
                    });
                    next_seq = next_seq.wrapping_add(1);
                }
                _ => break,
            }

            cursor += 1;
            if cursor >= maxlen {
                cursor = first;
            }
        }

        let last_sequence = if txns.is_empty() {
            sb.sequence.wrapping_sub(1)
        } else {
            txns.last().unwrap().sequence
        };
        Ok(ScanResult { txns, last_sequence })
    }
}
