//! Journal (jbd2) engine.
//!
//! Task 1.1: load the journal from inode 8 and map journal-file logical blocks
//! to their physical fs blocks.

use crate::prelude::*;
use crate::ext4_defs::*;
use crate::return_errno_with_message;

mod transaction;
pub use transaction::*;

mod device;
pub use device::*;

// Re-export the jbd2 on-disk codec (defined in `ext4_defs::journal`) so it is
// reachable from the crate root alongside the `Journal` engine. Recovery and the
// test harness assemble/parse/verify log blocks with these.
pub use crate::ext4_defs::journal::{
    assemble_descriptor_block, descriptor_block_tag_capacity, finalize_commit_csum,
    finalize_revoke_csum, jbd2_block_csum, jbd2_csum_seed, jbd2_data_block_csum,
    parse_descriptor_block, patch_journal_sb_head, verify_commit_csum, verify_revoke_csum,
    BlockTag, CommitBlock, JournalSuperblock, RevokeBlock, TagFormat, JBD2_COMMIT_BLOCK,
    JBD2_DESCRIPTOR_BLOCK, JBD2_FEATURE_INCOMPAT_64BIT, JBD2_FEATURE_INCOMPAT_CSUM_V2,
    JBD2_FEATURE_INCOMPAT_CSUM_V3, JBD2_FEATURE_INCOMPAT_REVOKE, JBD2_FLAG_DELETED,
    JBD2_FLAG_ESCAPE, JBD2_FLAG_LAST_TAG, JBD2_FLAG_SAME_UUID, JBD2_MAGIC_NUMBER,
    JBD2_REVOKE_BLOCK,
};

/// Where to simulate a crash during commit, for crash-recovery testing.
/// `None` is a full commit. `AfterCommitBlock` stops right after the commit
/// block is durable (steps 1-6), leaving the journal dirty with a committed but
/// un-checkpointed transaction — exactly what recovery must replay.
/// `MidCheckpoint(n)` checkpoints only the first `n` blocks then stops (for
/// checkpoint-idempotency testing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashPoint {
    None,
    AfterCommitBlock,
    MidCheckpoint(usize),
}

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

/// The LIVE on-disk journal superblock plus the format/feature context derived
/// from it. Recovery and commit both anchor on the live sb, so these fields are
/// the single source of truth for "what format is this journal".
struct LiveSb {
    sb: JournalSuperblock,
    fmt: TagFormat,
    has_64bit: bool,
    has_csum: bool,
    seed: u32,
}

impl Journal {
    /// Read the LIVE on-disk journal superblock (log block 0) and derive the
    /// format/feature context. Recovery and commit BOTH anchor on the live sb
    /// (not the cached `self.sb`), so this is the single source of truth for the
    /// "what format is this journal" question.
    fn live_sb(&self, fs: &Ext4) -> Result<LiveSb> {
        let sb_block = self.read_log_block(fs, 0)?;
        let sb = JournalSuperblock::parse(&sb_block)?;
        let fmt = TagFormat::from_features(sb.feature_incompat);
        let has_64bit = sb.feature_incompat & JBD2_FEATURE_INCOMPAT_64BIT != 0;
        let has_csum = sb.feature_incompat
            & (JBD2_FEATURE_INCOMPAT_CSUM_V2 | JBD2_FEATURE_INCOMPAT_CSUM_V3)
            != 0;
        let seed = jbd2_csum_seed(&sb.uuid);
        Ok(LiveSb {
            sb,
            fmt,
            has_64bit,
            has_csum,
            seed,
        })
    }

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

    /// The live on-disk journal sequence (s_sequence), read fresh from log block
    /// 0. This is the sequence the next transaction must commit at; the running
    /// transaction is begun with it so the log we emit matches what recovery
    /// expects.
    pub fn live_sequence(&self, fs: &Ext4) -> Result<u32> {
        Ok(self.live_sb(fs)?.sb.sequence)
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
        let LiveSb {
            sb,
            fmt,
            has_64bit,
            has_csum,
            seed,
        } = self.live_sb(fs)?;

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

    /// Commit a transaction: write it to the journal log, then checkpoint it to the
    /// blocks' final on-disk locations, leaving the journal clean. Crash-safe
    /// ordering (see comments). No-op for an empty transaction.
    ///
    /// LIMITATION: this assumes every block tag fits in ONE descriptor block.
    /// Multi-descriptor transactions are a future enhancement; typical metadata
    /// transactions are small. If the tags would overflow one block, commit
    /// returns `ENOSPC` rather than silently corrupting the log.
    pub fn commit(&self, fs: &Ext4, txn: &Transaction) -> Result<()> {
        self.commit_with_crash(fs, txn, CrashPoint::None)
    }

    /// Commit `txn`, optionally simulating a crash partway through (for crash-
    /// recovery tests). `CrashPoint::None` is a full commit (steps 1-8) and is
    /// exactly what `commit` does. `AfterCommitBlock` runs steps 1-6 then stops,
    /// leaving the journal dirty with a committed-but-un-checkpointed txn —
    /// precisely what recovery must replay. `MidCheckpoint(n)` checkpoints only
    /// the first `n` blocks then stops, leaving the journal still dirty (for
    /// idempotency testing).
    pub fn commit_with_crash(
        &self,
        fs: &Ext4,
        txn: &Transaction,
        crash: CrashPoint,
    ) -> Result<()> {
        // 1. Empty transaction → nothing to do.
        if txn.is_empty() {
            return Ok(());
        }

        // 2. Re-read the LIVE on-disk superblock (log block 0) and derive the
        // feature flags / csum seed exactly as `scan`/`build_revoke_table` do, so
        // the log we emit matches what recovery will parse.
        let ctx = self.live_sb(fs)?;
        let fmt = ctx.fmt;
        let has_64bit = ctx.has_64bit;
        let has_csum = ctx.has_csum;
        let seed = ctx.seed;
        let bs = fs.block_size();
        let first = ctx.sb.first;
        let maxlen = ctx.sb.maxlen;
        let seq = txn.sequence;

        // 2b. Build the set of blocks to journal + checkpoint. This is the staged
        // blocks plus a guaranteed copy of the ext4-superblock block with the
        // RECOVER incompat flag SET (and a corrected sb checksum). The superblock
        // is itself a journaled block: it gets CHECKPOINTED to its final location,
        // and a MidCheckpoint crash can write it (a low block number) before the
        // journal is cleared. If the staged sb had RECOVER clear, that crash would
        // leave RECOVER clear on disk while the journal is still dirty (a kernel
        // would NOT auto-recover). So we patch RECOVER ON in the staged sb copy,
        // adding it from disk if this txn didn't already stage it.
        let sb_block_idx = (1024 / bs) as u64;
        let sb_off = 1024 % bs;
        // BTreeMap keeps the journal/checkpoint order deterministic (sorted by
        // final block number), matching the previous direct iteration of txn.blocks.
        let mut journaled: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        for (final_block, staged) in &txn.blocks {
            journaled.insert(*final_block, staged.data.clone());
        }
        {
            let buf = journaled.entry(sb_block_idx).or_insert_with(|| {
                fs.block_device.read_offset(sb_block_idx as usize * bs, bs)
            });
            Ext4Superblock::patch_recover_in_block(buf, sb_off, true);
        }

        // A log cursor that wraps back to `first` once it reaches the end of the
        // journal area, mirroring the SCAN walk.
        // this engine commits one txn at a time; the log always starts at s_first
        let mut cursor = first;
        let mut advance = |c: &mut u32| {
            *c += 1;
            if *c >= maxlen {
                *c = first;
            }
        };

        // 3a. Build the descriptor block (single-descriptor limit). For each staged
        // block (BTreeMap → sorted by final block number) build a tag, handling the
        // ESCAPE case (logged copy zeroes the leading magic; tag flags ESCAPE).
        let magic_be = JBD2_MAGIC_NUMBER.to_be_bytes();
        let mut tags: Vec<BlockTag> = Vec::with_capacity(journaled.len());
        // Logged copies are the (possibly escaped) block bodies we write to the log.
        let mut logged_copies: Vec<Vec<u8>> = Vec::with_capacity(journaled.len());
        for (final_block, data) in &journaled {
            let mut flags: u16 = 0;
            let mut logged = data.clone();
            if data.len() >= 4 && data[0..4] == magic_be[..] {
                flags |= JBD2_FLAG_ESCAPE;
                logged[0..4].copy_from_slice(&[0u8; 4]);
            }
            // Per-data-block tag checksum over the LOGGED (post-escape) copy. Only
            // meaningful under CSUM_V2/V3, but cheap to always compute.
            let checksum = if has_csum {
                jbd2_data_block_csum(seed, seq, &logged)
            } else {
                0
            };
            tags.push(BlockTag {
                blocknr: *final_block,
                flags,
                checksum,
            });
            logged_copies.push(logged);
        }

        // Single-descriptor guard: format-aware via the shared codec helper, which
        // accounts for the header, first-tag UUID, per-format tag size, and tail
        // checksum. If the tags would exceed one block, bail loudly.
        if tags.len() > descriptor_block_tag_capacity(bs, fmt, has_64bit, has_csum) {
            return_errno_with_message!(
                Errno::ENOSPC,
                "txn too large for one descriptor block"
            );
        }

        let descriptor =
            assemble_descriptor_block(seed, seq, bs, fmt, has_64bit, has_csum, &tags);
        self.write_log_block(fs, cursor, &descriptor)?;
        advance(&mut cursor);

        // 3b. Data blocks: write each logged (escaped) copy in the SAME order.
        for logged in &logged_copies {
            self.write_log_block(fs, cursor, logged)?;
            advance(&mut cursor);
        }

        // 3c. Revoke block (only when the txn revokes something).
        if !txn.revokes.is_empty() {
            let mut blk = RevokeBlock {
                sequence: seq,
                blocks: txn.revokes.iter().copied().collect(),
            }
            .emit(bs, has_64bit);
            if has_csum {
                finalize_revoke_csum(&mut blk, seed);
            }
            self.write_log_block(fs, cursor, &blk)?;
            advance(&mut cursor);
        }

        // 4. Make the log body durable BEFORE the commit block.
        fs.block_device.flush();

        // 5. Mark the log active (s_start=first, s_sequence=seq) and flush, BEFORE
        // the commit becomes durable. A crash here → recovery scans from `first`,
        // finds the descriptor but no valid commit → discards the partial txn.
        // Safe: commit() hasn't returned, so losing this txn is fine.
        // re-read + patch in place to preserve sb fields the parser doesn't model (s_checksum, etc.); see recover()
        let mut sb_block = self.read_log_block(fs, 0)?;
        patch_journal_sb_head(&mut sb_block, seq, first);
        self.write_log_block(fs, 0, &sb_block)?;
        fs.block_device.flush();

        // 6. Commit block — terminates the transaction in the log. After this flush
        // the txn is durably committed; a crash now → recovery replays it.
        let mut commit_blk = CommitBlock {
            sequence: seq,
            commit_sec: 0, // timestamps are not used by recovery
            commit_nsec: 0,
        }
        .emit(bs);
        if has_csum {
            finalize_commit_csum(&mut commit_blk, seed);
        }
        self.write_log_block(fs, cursor, &commit_blk)?;
        advance(&mut cursor);
        fs.block_device.flush();

        // 6b. The journal is now durably dirty (a committed, un-checkpointed txn).
        // Set the ext4 RECOVER incompat flag on the LIVE on-disk superblock so any
        // crash in the commit→checkpoint window leaves RECOVER set on disk — a
        // kernel/e2fsck then knows the image needs recovery (without the
        // "needs_recovery flag is clear" warning). The staged sb copy (patched
        // above) covers the MidCheckpoint case where the sb is checkpointed early;
        // this covers the AfterCommitBlock / pre-checkpoint case where the live sb
        // is otherwise never touched. Cleared in step 8b on a full commit, and by
        // recover() after replay.
        let mut ext4_sb = fs.read_super_block();
        ext4_sb.set_needs_recovery(true);
        ext4_sb.sync_to_disk_with_csum(&fs.block_device);
        fs.block_device.flush();

        // Crash injection: stop here, after the commit block is durable but before
        // any checkpoint. The journal is dirty with a committed txn → recovery must
        // replay it on the next mount.
        if crash == CrashPoint::AfterCommitBlock {
            return Ok(());
        }

        // 7. Checkpoint: write the ORIGINAL (un-escaped) data to each block's final
        // on-disk location, then flush so the checkpoint is durable before we mark
        // the journal clean.
        for (i, (final_block, data)) in journaled.iter().enumerate() {
            // Crash injection: checkpoint only the first `n` blocks then stop,
            // leaving the journal still dirty (the mark-clean in step 8 never runs).
            if let CrashPoint::MidCheckpoint(n) = crash {
                if i >= n {
                    return Ok(());
                }
            }
            fs.block_device
                .write_offset(*final_block as usize * bs, data);
        }
        fs.block_device.flush();

        // 8. Mark clean: s_start=0, s_sequence=seq+1. The checkpointed data is now
        // the authoritative on-disk state; the log can be reused.
        let mut sb_block = self.read_log_block(fs, 0)?;
        patch_journal_sb_head(&mut sb_block, seq.wrapping_add(1), 0);
        self.write_log_block(fs, 0, &sb_block)?;
        fs.block_device.flush();

        // 8b. Clear the ext4 RECOVER flag on the LIVE superblock: the journal is
        // now clean (the staged sb with RECOVER set was just checkpointed, so the
        // on-disk sb currently has RECOVER set + correct free counts). Reading the
        // live sb picks up those checkpointed free counts; we clear RECOVER and
        // sync with a fresh checksum, leaving a clean, consistent superblock.
        let mut ext4_sb = fs.read_super_block();
        ext4_sb.set_needs_recovery(false);
        ext4_sb.sync_to_disk_with_csum(&fs.block_device);
        fs.block_device.flush();

        // 9. Done.
        Ok(())
    }

    /// Build the revoke table from a scan: block number → highest sequence that
    /// revoked it. During REPLAY a logged block is skipped when revoke_seq >=
    /// txn_seq. Reads each committed txn's revoke blocks. With a checksummed
    /// journal, a revoke block failing its checksum ABORTS recovery (Err) —
    /// dropping a revoke risks clobbering live data, so we fail loud (per design).
    pub fn build_revoke_table(&self, fs: &Ext4, scan: &ScanResult) -> Result<BTreeMap<u64, u32>> {
        // Anchor feature flags / seed on the LIVE on-disk superblock, exactly as
        // `scan` does: re-read log block 0 and derive has_64bit/has_csum/seed.
        let LiveSb {
            has_64bit,
            has_csum,
            seed,
            ..
        } = self.live_sb(fs)?;

        let mut table: BTreeMap<u64, u32> = BTreeMap::new();
        for txn in &scan.txns {
            for &log_idx in &txn.revoke_log_blocks {
                let block = self.read_log_block(fs, log_idx as Ext4Lblk)?;
                if has_csum && !verify_revoke_csum(&block, seed) {
                    return_errno_with_message!(Errno::EIO, "revoke block checksum failed");
                }
                let rb = RevokeBlock::parse(&block, has_64bit)?;
                for b in rb.blocks {
                    // Keep the highest revoking sequence per block. Inserting the
                    // first-seen value (rather than 0) is robust for any seq.
                    table
                        .entry(b)
                        .and_modify(|s| *s = (*s).max(rb.sequence))
                        .or_insert(rb.sequence);
                }
            }
        }
        Ok(table)
    }

    /// Recover a dirty journal: replay all committed transactions to their final
    /// on-disk locations (honoring revokes and the ESCAPE flag), then clear the
    /// journal so the next mount sees it clean. Idempotent: a second call after a
    /// successful recovery is a no-op (the journal is already clear → scan empty).
    pub fn recover(&self, fs: &Ext4) -> Result<()> {
        let scan = self.scan(fs)?;
        if scan.txns.is_empty() {
            return Ok(()); // clean journal, nothing to replay
        }
        let revokes = self.build_revoke_table(fs, &scan)?;
        let bs = fs.block_size();
        for txn in &scan.txns {
            for lb in &txn.blocks {
                // Skip a block revoked by this-or-a-later transaction.
                if let Some(&rseq) = revokes.get(&lb.final_block) {
                    if rseq >= txn.sequence {
                        continue;
                    }
                }
                let mut data = self.read_log_block(fs, lb.data_log_block as Ext4Lblk)?;
                // Honor ESCAPE: the logged copy of a block that began with the jbd2
                // magic has its first 4 bytes zeroed; restore the magic on replay.
                if lb.flags & JBD2_FLAG_ESCAPE != 0 {
                    data[0..4].copy_from_slice(&JBD2_MAGIC_NUMBER.to_be_bytes());
                }
                fs.block_device.write_offset(lb.final_block as usize * bs, &data);
            }
        }
        fs.block_device.flush();
        // Clear the journal: re-read the live superblock block, set s_start=0 and
        // s_sequence = last_committed + 1, write it back, flush. We patch the two
        // BE fields in place (rather than re-emit) to preserve fields the parser
        // doesn't model — s_checksum, s_errno, s_nr_users, … — which a full
        // emit() would zero, corrupting the superblock.
        let mut sb_block = self.read_log_block(fs, 0)?;
        patch_journal_sb_head(&mut sb_block, scan.last_sequence.wrapping_add(1), 0);
        self.write_log_block(fs, 0, &sb_block)?;
        fs.block_device.flush();

        // The image is now recovered and clean: clear the ext4 RECOVER incompat
        // flag on the live superblock (replay may have checkpointed a sb with
        // RECOVER set) and re-checksum it. Reading the live sb first preserves the
        // replayed free counts.
        let mut ext4_sb = fs.read_super_block();
        ext4_sb.set_needs_recovery(false);
        ext4_sb.sync_to_disk_with_csum(&fs.block_device);
        fs.block_device.flush();
        Ok(())
    }
}
