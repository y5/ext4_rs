//! In-memory write-side transaction model (jbd2 running transaction).
//!
//! Task 3.1: the accumulator that the write side uses to gather block writes
//! (and revokes) before committing them atomically (descriptor → data → commit →
//! checkpoint, built in Phase 4). Pure in-memory types — no fs/device here.

use crate::prelude::*; // BTreeMap, BTreeSet, Vec

/// One block staged for journaling, destined for its final on-disk location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedBlock {
    pub data: Vec<u8>,
    /// True if this block is filesystem metadata (vs file data).
    ///
    /// NOTE: in this `data=journaled` engine every block is journaled regardless,
    /// and jbd2 descriptor tags carry no on-disk metadata flag — so this field is
    /// currently always `false` and not yet load-bearing. It is retained for a
    /// possible future `data=ordered` mode (where only metadata would be journaled).
    pub is_metadata: bool,
}

/// A running journal transaction: the set of block writes (and revokes) that
/// will be committed atomically. Keyed by final fs block number; the last write
/// to a given block within the transaction wins (jbd2 coalescing).
#[derive(Debug, Clone)]
pub struct Transaction {
    pub sequence: u32,
    pub blocks: BTreeMap<u64, StagedBlock>,
    pub revokes: BTreeSet<u64>,
}

impl Transaction {
    pub fn new(sequence: u32) -> Self {
        Transaction {
            sequence,
            blocks: BTreeMap::new(),
            revokes: BTreeSet::new(),
        }
    }

    /// Stage a write of `data` destined for final fs block `final_block`.
    /// Overwrites any earlier staged write to the same block (last wins).
    pub fn stage(&mut self, final_block: u64, data: Vec<u8>, is_metadata: bool) {
        self.blocks
            .insert(final_block, StagedBlock { data, is_metadata });
    }

    /// Mark `block` as revoked in this transaction.
    pub fn revoke(&mut self, block: u64) {
        self.revokes.insert(block);
    }

    /// True if nothing has been staged (no blocks, no revokes).
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty() && self.revokes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_dedups_last_write_wins() {
        let mut t = Transaction::new(5);
        t.stage(100, vec![1u8; 10], false);
        t.stage(100, vec![2u8; 10], true); // same block → overwrites
        t.stage(101, vec![3u8; 10], false);
        assert_eq!(t.blocks.len(), 2);
        assert_eq!(t.blocks[&100].data, vec![2u8; 10]);
        assert!(t.blocks[&100].is_metadata);
        assert_eq!(t.blocks[&101].data, vec![3u8; 10]);
    }

    #[test]
    fn revoke_recorded_and_is_empty_tracks_state() {
        let mut t = Transaction::new(7);
        assert!(t.is_empty());
        t.revoke(200);
        assert!(!t.is_empty());
        assert!(t.revokes.contains(&200));
        t.revoke(200); // idempotent
        assert_eq!(t.revokes.len(), 1);
    }
}
