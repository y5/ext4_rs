//! Extended attributes (xattr).
//!
//! Attributes live in one of two places, and a file may use both:
//!   * **in-inode (ibody)** — in the spare bytes of the inode record, right
//!     after the fixed fields + `i_extra_isize`, introduced by a 4-byte magic.
//!   * **external block** — a single block pointed to by `i_file_acl`, with a
//!     32-byte header (magic, refcount, blocks, hash, checksum).
//!
//! In both layouts the entry array grows up from just after the header and the
//! values grow down from the end of the region; `e_value_offs` is measured from
//! the region base (the inode's IFIRST for ibody, the block start for a block).

use crate::ext4_defs::*;
use crate::prelude::*;
use crate::return_errno_with_message;

/// Identifies an xattr header (both ibody and block).
pub const EXT4_XATTR_MAGIC: u32 = 0xEA02_0000;
/// Fixed part of an `ext4_xattr_entry` (before the name).
const ENTRY_FIXED: usize = 16;
/// Entries and values are padded to this many bytes.
const XATTR_PAD: usize = 4;
/// Size of the external-block header.
const BLOCK_HDR: usize = 32;

/// On-disk size of an entry record: fixed part + name, padded.
fn entry_len(name_len: usize) -> usize {
    (ENTRY_FIXED + name_len + XATTR_PAD - 1) & !(XATTR_PAD - 1)
}

/// Map a name prefix to its on-disk `e_name_index` and the remaining suffix.
/// Longest / most specific prefixes are tried first.
fn split_name(full: &str) -> Option<(u8, &str)> {
    // Whole-name attributes (the prefix *is* the entire name).
    if full == "system.posix_acl_access" {
        return Some((2, ""));
    }
    if full == "system.posix_acl_default" {
        return Some((3, ""));
    }
    for (idx, prefix) in [(1u8, "user."), (4, "trusted."), (6, "security."), (7, "system.")] {
        if let Some(rest) = full.strip_prefix(prefix) {
            return Some((idx, rest));
        }
    }
    None
}

/// Reconstruct the full attribute name from an `e_name_index` and stored suffix.
fn join_name(name_index: u8, suffix: &[u8]) -> Option<String> {
    let prefix = match name_index {
        1 => "user.",
        2 => "system.posix_acl_access",
        3 => "system.posix_acl_default",
        4 => "trusted.",
        6 => "security.",
        7 => "system.",
        _ => return None,
    };
    let suffix = core::str::from_utf8(suffix).ok()?;
    Some(alloc::format!("{}{}", prefix, suffix))
}

/// A decoded attribute.
struct ParsedXattr {
    name_index: u8,
    name: Vec<u8>,
    value: Vec<u8>,
}

/// Parse the entry array at `entries`, resolving each value out of `value_base`
/// (the region `e_value_offs` is measured from). Stops at the terminating zero
/// entry or a malformed record.
fn parse_entries(entries: &[u8], value_base: &[u8]) -> Vec<ParsedXattr> {
    let mut out = Vec::new();
    let mut off = 0;
    while off + 4 <= entries.len() {
        let name_len = entries[off] as usize;
        let name_index = entries[off + 1];
        // Terminating entry: e_name_len == 0 && e_name_index == 0.
        if name_len == 0 && name_index == 0 {
            break;
        }
        if off + ENTRY_FIXED + name_len > entries.len() {
            break;
        }
        let e_value_offs = u16::from_le_bytes([entries[off + 2], entries[off + 3]]) as usize;
        let e_value_inum =
            u32::from_le_bytes([entries[off + 4], entries[off + 5], entries[off + 6], entries[off + 7]]);
        let e_value_size = u32::from_le_bytes([
            entries[off + 8],
            entries[off + 9],
            entries[off + 10],
            entries[off + 11],
        ]) as usize;

        let name = entries[off + ENTRY_FIXED..off + ENTRY_FIXED + name_len].to_vec();

        // A value stored in a separate inode (e_value_inum != 0) is not
        // supported; report it as empty rather than reading garbage.
        let value = if e_value_inum == 0 && e_value_offs + e_value_size <= value_base.len() {
            value_base[e_value_offs..e_value_offs + e_value_size].to_vec()
        } else {
            Vec::new()
        };

        out.push(ParsedXattr {
            name_index,
            name,
            value,
        });
        off += entry_len(name_len);
    }
    out
}

impl Ext4 {
    /// Physical block holding this inode's external xattrs, or 0 if none.
    fn xattr_block_of(&self, inode: &Ext4Inode) -> u64 {
        inode.file_acl as u64 | ((inode.osd2.l_i_file_acl_high as u64) << 32)
    }

    /// Decode all attributes of an inode, from both the inode body and any
    /// external block.
    fn xattr_collect(&self, ino: u32) -> Vec<ParsedXattr> {
        let mut out = Vec::new();
        let inode_ref = self.get_inode_ref(ino);
        let inode_size = self.super_block.inode_size() as usize;

        // In-inode (ibody) attributes.
        let extra = inode_ref.inode.i_extra_isize() as usize;
        let ibody_off = EXT4_GOOD_OLD_INODE_SIZE as usize + extra;
        if extra > 0 && ibody_off + 4 <= inode_size {
            let record = self
                .block_device
                .read_offset(self.inode_disk_pos(ino), inode_size);
            let magic = u32::from_le_bytes([
                record[ibody_off],
                record[ibody_off + 1],
                record[ibody_off + 2],
                record[ibody_off + 3],
            ]);
            if magic == EXT4_XATTR_MAGIC {
                let region = &record[ibody_off + 4..inode_size];
                out.extend(parse_entries(region, region));
            }
        }

        // External block attributes.
        let blk = self.xattr_block_of(&inode_ref.inode);
        if blk != 0 {
            let bs = self.block_size();
            let data = self.block_device.read_offset(blk as usize * bs, bs);
            let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            if magic == EXT4_XATTR_MAGIC {
                out.extend(parse_entries(&data[BLOCK_HDR..], &data));
            }
        }

        out
    }

    /// Get one attribute's value, or `ENODATA` if it is not present.
    pub fn xattr_get(&self, ino: u32, name: &str) -> Result<Vec<u8>> {
        let (idx, suffix) = match split_name(name) {
            Some(v) => v,
            None => return_errno_with_message!(Errno::ENODATA, "unsupported xattr namespace"),
        };
        for attr in self.xattr_collect(ino) {
            if attr.name_index == idx && attr.name == suffix.as_bytes() {
                return Ok(attr.value);
            }
        }
        return_errno_with_message!(Errno::ENODATA, "no such xattr")
    }

    /// List all attribute names of an inode.
    pub fn xattr_list(&self, ino: u32) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for attr in self.xattr_collect(ino) {
            if let Some(full) = join_name(attr.name_index, &attr.name) {
                names.push(full);
            }
        }
        Ok(names)
    }
}
