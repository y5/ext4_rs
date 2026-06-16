//! jbd2 (journaling block device v2) on-disk format.
//!
//! NOTE: unlike ext4 metadata, jbd2 structures are stored BIG-ENDIAN on disk.
//! Every codec here byte-swaps explicitly.

use crate::prelude::*;
use crate::return_errno_with_message;

pub const JBD2_MAGIC_NUMBER: u32 = 0xC03B3998;

// Block types (journal_header_t.h_blocktype)
pub const JBD2_DESCRIPTOR_BLOCK: u32 = 1;
pub const JBD2_COMMIT_BLOCK: u32 = 2;
pub const JBD2_SUPERBLOCK_V1: u32 = 3;
pub const JBD2_SUPERBLOCK_V2: u32 = 4;
pub const JBD2_REVOKE_BLOCK: u32 = 5;

// feature_incompat bits (journal superblock)
pub const JBD2_FEATURE_INCOMPAT_REVOKE: u32 = 0x0001;
pub const JBD2_FEATURE_INCOMPAT_64BIT: u32 = 0x0002;
pub const JBD2_FEATURE_INCOMPAT_ASYNC_COMMIT: u32 = 0x0004;
pub const JBD2_FEATURE_INCOMPAT_CSUM_V2: u32 = 0x0008;
pub const JBD2_FEATURE_INCOMPAT_CSUM_V3: u32 = 0x0010;

// block-tag flags (journal_block_tag_t.t_flags)
pub const JBD2_FLAG_ESCAPE: u16 = 1;     // block began with the magic, was escaped
pub const JBD2_FLAG_SAME_UUID: u16 = 2;  // no UUID field follows this tag
pub const JBD2_FLAG_DELETED: u16 = 4;
pub const JBD2_FLAG_LAST_TAG: u16 = 8;   // last tag in this descriptor block

/// Minimum number of bytes we must be able to read to parse the fields below.
/// The parsed fields span the first ~81 bytes (`s_checksum_type` is read at
/// offset 80). We require 1024 bytes — the minimum sane on-disk journal block
/// size — as the minimum, guarding against truncated blocks.
const JBD2_SUPERBLOCK_MIN_LEN: usize = 1024;

/// Parsed jbd2 journal superblock (`journal_superblock_t`).
///
/// Fields are OWNED, decoded big-endian from the on-disk block. We deliberately
/// avoid `transmute`/`#[repr(C)]` overlay because jbd2 is big-endian whereas the
/// host (and the rest of this crate) is little-endian.
///
/// On-disk byte offsets (from the start of the block) of the fields we decode:
///   0  h_magic       (u32 BE) — must equal `JBD2_MAGIC_NUMBER`
///   4  h_blocktype   (u32 BE)
///   8  h_sequence    (u32 BE) — header sequence (unused here)
///   12 s_blocksize   (u32 BE)
///   16 s_maxlen      (u32 BE)
///   20 s_first       (u32 BE)
///   24 s_sequence    (u32 BE) — first commit ID expected in the log
///   28 s_start       (u32 BE) — start-of-log block; 0 == clean/empty
///   36 s_feature_compat    (u32 BE)
///   40 s_feature_incompat  (u32 BE)
///   44 s_feature_ro_compat (u32 BE)
///   48 s_uuid[16]    (raw bytes)
///   80 s_checksum_type (u8)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JournalSuperblock {
    /// h_blocktype — expected to be `JBD2_SUPERBLOCK_V2` (or `_V1`).
    pub blocktype: u32,
    /// s_blocksize — journal block size in bytes.
    pub blocksize: u32,
    /// s_maxlen — total number of blocks in the journal.
    pub maxlen: u32,
    /// s_first — first block of log information (after the superblock).
    pub first: u32,
    /// s_sequence — first commit ID expected in the log.
    pub sequence: u32,
    /// s_start — block number of the start of log; 0 means clean/empty.
    pub start: u32,
    /// s_feature_compat.
    pub feature_compat: u32,
    /// s_feature_incompat.
    pub feature_incompat: u32,
    /// s_feature_ro_compat.
    pub feature_ro_compat: u32,
    /// s_uuid — journal UUID (raw 16 bytes).
    pub uuid: [u8; 16],
    /// s_checksum_type.
    pub checksum_type: u8,
}

impl JournalSuperblock {
    /// Decode a jbd2 journal superblock from a raw block buffer (big-endian).
    ///
    /// Returns `Errno::EINVAL` if the buffer is too short or `h_magic` does not
    /// match `JBD2_MAGIC_NUMBER`.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < JBD2_SUPERBLOCK_MIN_LEN {
            return_errno_with_message!(Errno::EINVAL, "journal superblock buffer too short");
        }

        let be32 = |off: usize| -> u32 {
            u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        };

        let magic = be32(0);
        if magic != JBD2_MAGIC_NUMBER {
            return_errno_with_message!(Errno::EINVAL, "bad jbd2 journal superblock magic");
        }

        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&buf[48..64]);

        Ok(JournalSuperblock {
            blocktype: be32(4),
            blocksize: be32(12),
            maxlen: be32(16),
            first: be32(20),
            sequence: be32(24),
            start: be32(28),
            feature_compat: be32(36),
            feature_incompat: be32(40),
            feature_ro_compat: be32(44),
            uuid,
            checksum_type: buf[80],
        })
    }

    /// Encode this superblock into a fresh, zeroed journal block (big-endian).
    ///
    /// Mirrors [`Self::parse`] offset-for-offset. The returned buffer is at least
    /// one block long: `max(self.blocksize, JBD2_SUPERBLOCK_MIN_LEN)` bytes, so it
    /// can be written directly back to the journal's superblock block. The
    /// checksum field (`s_checksum`) is left zero; checksum computation is the
    /// caller's / a later task's responsibility.
    pub fn emit(&self) -> Vec<u8> {
        let len = core::cmp::max(self.blocksize as usize, JBD2_SUPERBLOCK_MIN_LEN);
        let mut buf = vec![0u8; len];

        let put32 = |buf: &mut [u8], off: usize, v: u32| {
            buf[off..off + 4].copy_from_slice(&v.to_be_bytes());
        };

        put32(&mut buf, 0, JBD2_MAGIC_NUMBER); // h_magic
        put32(&mut buf, 4, self.blocktype); // h_blocktype
        put32(&mut buf, 12, self.blocksize); // s_blocksize
        put32(&mut buf, 16, self.maxlen); // s_maxlen
        put32(&mut buf, 20, self.first); // s_first
        put32(&mut buf, 24, self.sequence); // s_sequence
        put32(&mut buf, 28, self.start); // s_start
        put32(&mut buf, 36, self.feature_compat); // s_feature_compat
        put32(&mut buf, 40, self.feature_incompat); // s_feature_incompat
        put32(&mut buf, 44, self.feature_ro_compat); // s_feature_ro_compat
        buf[48..64].copy_from_slice(&self.uuid); // s_uuid
        buf[80] = self.checksum_type; // s_checksum_type

        buf
    }

    /// True if the given `JBD2_FEATURE_INCOMPAT_*` bit is set.
    pub fn has_incompat(&self, bit: u32) -> bool {
        self.feature_incompat & bit != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_superblock_parses_be_fields() {
        let mut b = vec![0u8; 1024];
        b[0..4].copy_from_slice(&JBD2_MAGIC_NUMBER.to_be_bytes());
        b[4..8].copy_from_slice(&JBD2_SUPERBLOCK_V2.to_be_bytes());
        b[12..16].copy_from_slice(&4096u32.to_be_bytes());   // s_blocksize
        b[16..20].copy_from_slice(&1024u32.to_be_bytes());   // s_maxlen
        b[20..24].copy_from_slice(&1u32.to_be_bytes());      // s_first
        b[24..28].copy_from_slice(&7u32.to_be_bytes());      // s_sequence
        b[28..32].copy_from_slice(&3u32.to_be_bytes());      // s_start
        b[40..44].copy_from_slice(
            &(JBD2_FEATURE_INCOMPAT_REVOKE | JBD2_FEATURE_INCOMPAT_64BIT | JBD2_FEATURE_INCOMPAT_CSUM_V3).to_be_bytes());
        let sb = JournalSuperblock::parse(&b).unwrap();
        assert_eq!(sb.blocksize, 4096);
        assert_eq!(sb.maxlen, 1024);
        assert_eq!(sb.first, 1);
        assert_eq!(sb.sequence, 7);
        assert_eq!(sb.start, 3);
        assert!(sb.has_incompat(JBD2_FEATURE_INCOMPAT_CSUM_V3));
        assert!(!sb.has_incompat(JBD2_FEATURE_INCOMPAT_CSUM_V2));
    }

    #[test]
    fn journal_superblock_emit_roundtrips() {
        // Build an sb by parsing a hand-made block, emit it, re-parse, assert equal.
        let mut b = vec![0u8; 1024];
        b[0..4].copy_from_slice(&JBD2_MAGIC_NUMBER.to_be_bytes());
        b[4..8].copy_from_slice(&JBD2_SUPERBLOCK_V2.to_be_bytes());
        b[12..16].copy_from_slice(&2048u32.to_be_bytes());
        b[16..20].copy_from_slice(&512u32.to_be_bytes());
        b[20..24].copy_from_slice(&1u32.to_be_bytes());
        b[24..28].copy_from_slice(&42u32.to_be_bytes());
        b[28..32].copy_from_slice(&9u32.to_be_bytes());
        b[36..40].copy_from_slice(&0u32.to_be_bytes());
        b[40..44].copy_from_slice(&(JBD2_FEATURE_INCOMPAT_REVOKE | JBD2_FEATURE_INCOMPAT_CSUM_V3).to_be_bytes());
        b[44..48].copy_from_slice(&0u32.to_be_bytes());
        let uuid = [0x11u8; 16];
        b[48..64].copy_from_slice(&uuid);
        b[80] = 4; // checksum_type
        let sb = JournalSuperblock::parse(&b).unwrap();

        let out = sb.emit();
        // Emitted bytes must be at least one block and re-parse to an equal sb.
        let sb2 = JournalSuperblock::parse(&out).unwrap();
        assert_eq!(sb2.blocksize, 2048);
        assert_eq!(sb2.maxlen, 512);
        assert_eq!(sb2.first, 1);
        assert_eq!(sb2.sequence, 42);
        assert_eq!(sb2.start, 9);
        assert_eq!(sb2.feature_incompat, JBD2_FEATURE_INCOMPAT_REVOKE | JBD2_FEATURE_INCOMPAT_CSUM_V3);
        assert_eq!(sb2.uuid, uuid);
        assert_eq!(sb2.checksum_type, 4);
        // Magic + blocktype land big-endian at 0 / 4.
        assert_eq!(&out[0..4], &JBD2_MAGIC_NUMBER.to_be_bytes());
        assert_eq!(&out[4..8], &JBD2_SUPERBLOCK_V2.to_be_bytes());
    }

    #[test]
    fn journal_superblock_rejects_bad_magic() {
        let b = vec![0u8; 1024];               // all-zero == wrong magic
        assert!(JournalSuperblock::parse(&b).is_err());
    }

    #[test]
    fn journal_superblock_rejects_short_buffer() {
        // Valid magic but a buffer shorter than the minimum: the length guard must
        // reject it (this is what keeps the field reads panic-free).
        let mut b = vec![0u8; 64];
        b[0..4].copy_from_slice(&JBD2_MAGIC_NUMBER.to_be_bytes());
        assert!(JournalSuperblock::parse(&b).is_err());
    }
}
