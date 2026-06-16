//! jbd2 (journaling block device v2) on-disk format.
//!
//! NOTE: unlike ext4 metadata, jbd2 structures are stored BIG-ENDIAN on disk.
//! Every codec here byte-swaps explicitly.

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
