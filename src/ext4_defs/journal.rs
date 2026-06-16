//! jbd2 (journaling block device v2) on-disk format.
//!
//! NOTE: unlike ext4 metadata, jbd2 structures are stored BIG-ENDIAN on disk.
//! Every codec here byte-swaps explicitly.

use crate::prelude::*;
use crate::return_errno_with_message;

pub const JBD2_MAGIC_NUMBER: u32 = 0xC03B3998;

/// Length in bytes of a jbd2 UUID field (`s_uuid` / a block-tag's trailing UUID).
const JBD2_UUID_LEN: usize = 16;

// Big-endian read/write helpers shared by every codec in this module. jbd2 is
// big-endian on disk; these centralize the byte-swapping so the codecs read
// uniformly. Callers are responsible for length-guarding before calling.
fn be16(buf: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([buf[off], buf[off + 1]])
}

fn be32(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn be64(buf: &[u8], off: usize) -> u64 {
    u64::from_be_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

fn put_be16(buf: &mut [u8], off: usize, v: u16) {
    buf[off..off + 2].copy_from_slice(&v.to_be_bytes());
}

fn put_be32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_be_bytes());
}

fn put_be64(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_be_bytes());
}

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

        let magic = be32(buf, 0);
        if magic != JBD2_MAGIC_NUMBER {
            return_errno_with_message!(Errno::EINVAL, "bad jbd2 journal superblock magic");
        }

        let mut uuid = [0u8; JBD2_UUID_LEN];
        uuid.copy_from_slice(&buf[48..48 + JBD2_UUID_LEN]);

        Ok(JournalSuperblock {
            blocktype: be32(buf, 4),
            blocksize: be32(buf, 12),
            maxlen: be32(buf, 16),
            first: be32(buf, 20),
            sequence: be32(buf, 24),
            start: be32(buf, 28),
            feature_compat: be32(buf, 36),
            feature_incompat: be32(buf, 40),
            feature_ro_compat: be32(buf, 44),
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

        put_be32(&mut buf, 0, JBD2_MAGIC_NUMBER); // h_magic
        put_be32(&mut buf, 4, self.blocktype); // h_blocktype
        put_be32(&mut buf, 12, self.blocksize); // s_blocksize
        put_be32(&mut buf, 16, self.maxlen); // s_maxlen
        put_be32(&mut buf, 20, self.first); // s_first
        put_be32(&mut buf, 24, self.sequence); // s_sequence
        put_be32(&mut buf, 28, self.start); // s_start
        put_be32(&mut buf, 36, self.feature_compat); // s_feature_compat
        put_be32(&mut buf, 40, self.feature_incompat); // s_feature_incompat
        put_be32(&mut buf, 44, self.feature_ro_compat); // s_feature_ro_compat
        buf[48..48 + JBD2_UUID_LEN].copy_from_slice(&self.uuid); // s_uuid
        buf[80] = self.checksum_type; // s_checksum_type

        buf
    }

    /// True if the given `JBD2_FEATURE_INCOMPAT_*` bit is set.
    pub fn has_incompat(&self, bit: u32) -> bool {
        self.feature_incompat & bit != 0
    }
}

/// Which on-disk block-tag layout a journal uses, selected by its incompat
/// feature flags.
///
/// - `V1` — `journal_block_tag_t` with a zero checksum field (true v1, no csum).
/// - `V2` — `journal_block_tag_t` carrying the low 16 bits of the tag checksum
///   (`JBD2_FEATURE_INCOMPAT_CSUM_V2`).
/// - `V3` — `journal_block_tag3_t`, a wider layout with a full 32-bit checksum
///   and an explicit 32-bit high-half block number
///   (`JBD2_FEATURE_INCOMPAT_CSUM_V3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagFormat {
    V1,
    V2,
    V3,
}

impl TagFormat {
    /// Pick the tag layout from the journal's incompat feature flags.
    ///
    /// CSUM_V3 takes precedence over CSUM_V2 (the kernel never sets both
    /// meaningfully, but if both bits appear, V3 is the wider/newer format).
    pub fn from_features(feature_incompat: u32) -> Self {
        if feature_incompat & JBD2_FEATURE_INCOMPAT_CSUM_V3 != 0 {
            TagFormat::V3
        } else if feature_incompat & JBD2_FEATURE_INCOMPAT_CSUM_V2 != 0 {
            TagFormat::V2
        } else {
            TagFormat::V1
        }
    }
}

/// A single journal block tag (`journal_block_tag_t` / `journal_block_tag3_t`).
///
/// Inside a DESCRIPTOR block, after the 12-byte journal header, comes a run of
/// these tags — one per data block that follows in the log. Each tag names the
/// final on-disk location of the next logged block.
///
/// This struct holds the *logical* contents, independent of the three on-disk
/// layouts; [`Self::emit`] / [`Self::parse`] handle the per-format field widths
/// and big-endian encoding.
///
/// ## Trailing UUID
/// On disk, each tag is *optionally* followed by a 16-byte UUID: it is present
/// unless the tag's flags carry [`JBD2_FLAG_SAME_UUID`]. In practice the journal
/// sets SAME_UUID on every tag except the first in a descriptor block, so only
/// the first tag carries a UUID. This codec models the UUID's *presence* for
/// length accounting only — it does not store or return the UUID bytes. On emit,
/// 16 zero bytes are written as a placeholder when `!same_uuid`; the real UUID is
/// filled in by the descriptor-block builder (Task 0.5+). On parse, the trailing
/// UUID is skipped (and counted) when SAME_UUID is absent in the parsed flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTag {
    /// Full final block number (low | high combined into one value).
    pub blocknr: u64,
    /// Logical flags: ESCAPE / SAME_UUID / DELETED / LAST_TAG.
    pub flags: u16,
    /// Tag checksum. V3 uses the full 32 bits; V2 uses only the low 16 bits;
    /// V1 has no checksum field on disk (always 0).
    pub checksum: u32,
}

impl BlockTag {
    /// Encode this tag into its on-disk bytes (big-endian), per `fmt`.
    ///
    /// Layouts (all big-endian):
    /// - **V3** (`journal_block_tag3_t`, 16 bytes):
    ///   `t_blocknr`(u32) @0, `t_flags`(u32) @4, `t_blocknr_high`(u32) @8,
    ///   `t_checksum`(u32) @12. The high half is always written (the kernel
    ///   writes 0 when not 64bit).
    /// - **V1/V2** (`journal_block_tag_t`, 8 bytes, +4 if 64bit):
    ///   `t_blocknr`(u32) @0, `t_checksum`(u16) @4, `t_flags`(u16) @6,
    ///   and `t_blocknr_high`(u32) @8 only when `has_64bit`. V1 writes checksum
    ///   0; V2 writes the low 16 bits of `self.checksum`.
    ///
    /// If `!same_uuid`, 16 placeholder (zero) UUID bytes are appended — see the
    /// struct docs for why the real UUID is deferred to the descriptor builder.
    pub fn emit(&self, fmt: TagFormat, has_64bit: bool, same_uuid: bool) -> Vec<u8> {
        let mut buf = Vec::new();

        let lo = (self.blocknr & 0xFFFF_FFFF) as u32;
        let hi = (self.blocknr >> 32) as u32;

        match fmt {
            TagFormat::V3 => {
                buf.extend_from_slice(&lo.to_be_bytes()); // t_blocknr      @0
                // V3 t_flags is 32-bit on disk; only the low bits are defined, so we narrow to u16.
                buf.extend_from_slice(&(self.flags as u32).to_be_bytes()); // t_flags @4
                buf.extend_from_slice(&hi.to_be_bytes()); // t_blocknr_high @8 (always)
                buf.extend_from_slice(&self.checksum.to_be_bytes()); // t_checksum @12
            }
            TagFormat::V1 | TagFormat::V2 => {
                let csum16: u16 = match fmt {
                    TagFormat::V1 => 0,
                    _ => (self.checksum & 0xFFFF) as u16,
                };
                buf.extend_from_slice(&lo.to_be_bytes()); // t_blocknr   @0
                buf.extend_from_slice(&csum16.to_be_bytes()); // t_checksum @4
                buf.extend_from_slice(&self.flags.to_be_bytes()); // t_flags @6
                if has_64bit {
                    buf.extend_from_slice(&hi.to_be_bytes()); // t_blocknr_high @8
                }
            }
        }

        if !same_uuid {
            // Placeholder UUID; the descriptor-block builder fills the real bytes.
            buf.extend_from_slice(&[0u8; JBD2_UUID_LEN]);
        }

        buf
    }

    /// Parse a tag at the start of `buf`, returning the decoded tag and the TOTAL
    /// number of bytes consumed — INCLUDING the trailing 16-byte UUID when the
    /// parsed flags lack [`JBD2_FLAG_SAME_UUID`].
    ///
    /// Returns `Errno::EINVAL` if `buf` is too short for the selected layout (or
    /// for the trailing UUID when one is expected), mirroring
    /// [`JournalSuperblock::parse`]'s length guard so we never index out of range.
    pub fn parse(buf: &[u8], fmt: TagFormat, has_64bit: bool) -> Result<(BlockTag, usize)> {
        // Base tag size (without trailing UUID).
        let tag_len = match fmt {
            TagFormat::V3 => 16,
            TagFormat::V1 | TagFormat::V2 => {
                if has_64bit {
                    12
                } else {
                    8
                }
            }
        };

        if buf.len() < tag_len {
            return_errno_with_message!(Errno::EINVAL, "journal block tag buffer too short");
        }

        let (blocknr, flags, checksum) = match fmt {
            TagFormat::V3 => {
                let lo = be32(buf, 0) as u64;
                // V3 t_flags is 32-bit on disk; only the low bits are defined, so we narrow to u16.
                let flags = be32(buf, 4) as u16;
                let hi = be32(buf, 8) as u64;
                let checksum = be32(buf, 12);
                ((hi << 32) | lo, flags, checksum)
            }
            TagFormat::V1 | TagFormat::V2 => {
                let lo = be32(buf, 0) as u64;
                let csum16 = be16(buf, 4);
                let flags = be16(buf, 6);
                let hi = if has_64bit { be32(buf, 8) as u64 } else { 0 };
                let checksum = match fmt {
                    TagFormat::V1 => 0,
                    _ => csum16 as u32,
                };
                ((hi << 32) | lo, flags, checksum)
            }
        };

        // A UUID trails the tag unless SAME_UUID is set in the parsed flags.
        let mut consumed = tag_len;
        if flags & JBD2_FLAG_SAME_UUID == 0 {
            if buf.len() < tag_len + JBD2_UUID_LEN {
                return_errno_with_message!(
                    Errno::EINVAL,
                    "journal block tag buffer too short for trailing UUID"
                );
            }
            consumed += JBD2_UUID_LEN;
        }

        Ok((
            BlockTag {
                blocknr,
                flags,
                checksum,
            },
            consumed,
        ))
    }
}

/// A jbd2 commit block (`struct commit_header`), big-endian on disk.
///
/// A commit block terminates a transaction in the log. On-disk layout:
///   0..12  journal_header_t (h_magic, h_blocktype=`JBD2_COMMIT_BLOCK`, h_sequence)
///   12     h_chksum_type  (u8)
///   13     h_chksum_size  (u8)
///   14..16 h_padding[2]
///   16..48 h_chksum[8]    (8×u32) — the block checksum for CSUM_V2/V3 lives in
///          h_chksum[0]@16; left ZERO here (checksums are a later task).
///   48..56 h_commit_sec   (u64 BE)
///   56..60 h_commit_nsec  (u32 BE)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitBlock {
    /// h_sequence — the transaction's commit ID.
    pub sequence: u32,
    /// h_commit_sec — commit timestamp, whole seconds.
    pub commit_sec: u64,
    /// h_commit_nsec — commit timestamp, nanosecond fraction.
    pub commit_nsec: u32,
}

impl CommitBlock {
    /// Encode this commit block into a fresh, zeroed `block_size`-byte block
    /// (big-endian). The checksum region (`h_chksum`) is left zero — a later
    /// task computes it.
    pub fn emit(&self, block_size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; block_size];
        put_be32(&mut buf, 0, JBD2_MAGIC_NUMBER); // h_magic
        put_be32(&mut buf, 4, JBD2_COMMIT_BLOCK); // h_blocktype
        put_be32(&mut buf, 8, self.sequence); // h_sequence
        // h_chksum_type / h_chksum_size / h_padding / h_chksum[8] left zero.
        put_be64(&mut buf, 48, self.commit_sec); // h_commit_sec
        put_be32(&mut buf, 56, self.commit_nsec); // h_commit_nsec
        buf
    }

    /// Decode a commit block from a raw block buffer (big-endian).
    ///
    /// Returns `Errno::EINVAL` if the buffer is too short, the magic is wrong, or
    /// the blocktype is not `JBD2_COMMIT_BLOCK`.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        // Need through h_commit_nsec@56..60.
        if buf.len() < 60 {
            return_errno_with_message!(Errno::EINVAL, "journal commit block buffer too short");
        }
        if be32(buf, 0) != JBD2_MAGIC_NUMBER {
            return_errno_with_message!(Errno::EINVAL, "bad jbd2 commit block magic");
        }
        if be32(buf, 4) != JBD2_COMMIT_BLOCK {
            return_errno_with_message!(Errno::EINVAL, "journal block is not a commit block");
        }
        Ok(CommitBlock {
            sequence: be32(buf, 8),
            commit_sec: be64(buf, 48),
            commit_nsec: be32(buf, 56),
        })
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

    #[test]
    fn block_tag_v3_roundtrip() {
        // V3, 64bit: blocknr exercises the high half, SAME_UUID so no trailing UUID,
        // and the full 32-bit checksum must survive the round-trip.
        let tag = BlockTag {
            blocknr: 0x1_2345_6789,
            flags: JBD2_FLAG_SAME_UUID,
            checksum: 0xDEAD_BEEF,
        };
        let bytes = tag.emit(TagFormat::V3, true, true);
        // V3 tag is 16 bytes, no UUID because SAME_UUID is set.
        assert_eq!(bytes.len(), 16);

        let (parsed, consumed) = BlockTag::parse(&bytes, TagFormat::V3, true).unwrap();
        assert_eq!(parsed, tag);
        assert_eq!(consumed, bytes.len());
        // The high half of the block number must survive.
        assert_eq!(parsed.blocknr >> 32, 0x1);
        assert_eq!(parsed.checksum, 0xDEAD_BEEF);
    }

    #[test]
    fn block_tag_v2_roundtrip() {
        // V2, no 64bit: 8-byte tag, SAME_UUID, only low 16 bits of checksum survive.
        let tag = BlockTag {
            blocknr: 0x4242,
            flags: JBD2_FLAG_SAME_UUID,
            checksum: 0xBEEF,
        };
        let bytes = tag.emit(TagFormat::V2, false, true);
        assert_eq!(bytes.len(), 8);

        let (parsed, consumed) = BlockTag::parse(&bytes, TagFormat::V2, false).unwrap();
        assert_eq!(parsed, tag);
        assert_eq!(consumed, bytes.len());
        assert_eq!(parsed.checksum, 0xBEEF);
    }

    #[test]
    fn block_tag_v1_roundtrip() {
        // V1, no 64bit: 8-byte tag. LAST_TAG (NOT same_uuid) so a 16-byte UUID trails.
        // The v1 checksum field is always 0 on disk, so keep BlockTag.checksum == 0.
        let tag = BlockTag {
            blocknr: 0x99,
            flags: JBD2_FLAG_LAST_TAG,
            checksum: 0,
        };
        let bytes = tag.emit(TagFormat::V1, false, false);
        assert_eq!(bytes.len(), 8 + 16);

        let (parsed, consumed) = BlockTag::parse(&bytes, TagFormat::V1, false).unwrap();
        assert_eq!(parsed, tag);
        assert_eq!(consumed, 8 + 16);
    }

    #[test]
    fn commit_block_roundtrip() {
        let cb = CommitBlock {
            sequence: 5,
            commit_sec: 0x1122334455,
            commit_nsec: 0x6677,
        };
        let bytes = cb.emit(1024);
        assert_eq!(bytes.len(), 1024);
        // header magic + blocktype land big-endian at 0 / 4.
        assert_eq!(&bytes[0..4], &JBD2_MAGIC_NUMBER.to_be_bytes());
        assert_eq!(&bytes[4..8], &JBD2_COMMIT_BLOCK.to_be_bytes());

        let parsed = CommitBlock::parse(&bytes).unwrap();
        assert_eq!(parsed, cb);
    }

    #[test]
    fn commit_block_rejects_wrong_blocktype() {
        let cb = CommitBlock {
            sequence: 1,
            commit_sec: 0,
            commit_nsec: 0,
        };
        let mut bytes = cb.emit(1024);
        // Overwrite h_blocktype@4 with a non-commit blocktype.
        bytes[4..8].copy_from_slice(&JBD2_REVOKE_BLOCK.to_be_bytes());
        assert!(CommitBlock::parse(&bytes).is_err());
    }

    #[test]
    fn block_tag_format_selection() {
        // V3 wins even if V2 is also set.
        assert_eq!(
            TagFormat::from_features(JBD2_FEATURE_INCOMPAT_CSUM_V3 | JBD2_FEATURE_INCOMPAT_CSUM_V2),
            TagFormat::V3
        );
        assert_eq!(
            TagFormat::from_features(JBD2_FEATURE_INCOMPAT_CSUM_V2),
            TagFormat::V2
        );
        assert_eq!(TagFormat::from_features(0), TagFormat::V1);
    }
}
