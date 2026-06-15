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
use crate::utils::*;

/// Identifies an xattr header (both ibody and block).
pub const EXT4_XATTR_MAGIC: u32 = 0xEA02_0000;
/// Fixed part of an `ext4_xattr_entry` (before the name).
const ENTRY_FIXED: usize = 16;
/// Entries and values are padded to this many bytes.
const XATTR_PAD: usize = 4;
/// Size of the external-block header.
const BLOCK_HDR: usize = 32;

/// s_feature_incompat bit: large xattr values live in dedicated inodes.
const EXT4_FEATURE_INCOMPAT_EA_INODE: u32 = 0x400;
/// i_flags bit marking an inode that holds an xattr value.
const EXT4_EA_INODE_FL: u32 = 0x0020_0000;

/// setxattr flags.
const XATTR_CREATE: i32 = 1; // fail if the attribute already exists
const XATTR_REPLACE: i32 = 2; // fail if the attribute does not exist

/// On-disk size of an entry record: fixed part + name, padded.
fn entry_len(name_len: usize) -> usize {
    (ENTRY_FIXED + name_len + XATTR_PAD - 1) & !(XATTR_PAD - 1)
}

/// Value storage is padded to `XATTR_PAD`.
fn value_pad(size: usize) -> usize {
    (size + XATTR_PAD - 1) & !(XATTR_PAD - 1)
}

/// Serialize `attrs` into an entry-array-plus-values region of `region_len`
/// bytes (entries grow up from the start, values grow down from the end,
/// `e_value_offs` measured from the region start). `e_hash` is left 0, which is
/// correct for the inode body. Returns None if the attributes don't fit.
fn serialize_region(attrs: &[ParsedXattr], region_len: usize) -> Option<Vec<u8>> {
    let mut region = vec![0u8; region_len];

    // ext4 keeps entries sorted by (name_index, name).
    let mut sorted: Vec<&ParsedXattr> = attrs.iter().collect();
    sorted.sort_by(|a, b| (a.name_index, &a.name).cmp(&(b.name_index, &b.name)));

    let mut entry_off = 0usize; // entries grow up
    let mut value_end = region_len; // values grow down

    for a in sorted {
        let elen = entry_len(a.name.len());
        let vpad = value_pad(a.value.len());
        // Need room for this entry, the 4-byte terminator, and the value.
        if entry_off + elen + 4 > value_end.saturating_sub(vpad) {
            return None;
        }

        let e_value_offs = if a.value.is_empty() {
            0u16
        } else {
            let voff = value_end - vpad;
            region[voff..voff + a.value.len()].copy_from_slice(&a.value);
            value_end = voff;
            voff as u16
        };

        region[entry_off] = a.name.len() as u8;
        region[entry_off + 1] = a.name_index;
        region[entry_off + 2..entry_off + 4].copy_from_slice(&e_value_offs.to_le_bytes());
        // e_value_inum (4) stays 0
        region[entry_off + 8..entry_off + 12]
            .copy_from_slice(&(a.value.len() as u32).to_le_bytes());
        // e_hash (4) stays 0
        region[entry_off + ENTRY_FIXED..entry_off + ENTRY_FIXED + a.name.len()]
            .copy_from_slice(&a.name);

        entry_off += elen;
    }
    // The terminating zero entry is already present (region is zero-filled).
    Some(region)
}

/// Entry hash stored in `e_hash`: fold in the name (one byte at a time), then
/// the value padded to a multiple of 4 and read as little-endian u32 words —
/// the algorithm e2fsck validates against. Attribute names are ASCII, so the
/// historical signed-char ambiguity in the name loop doesn't arise.
fn hash_entry(name: &[u8], value: &[u8]) -> u32 {
    const NAME_SHIFT: u32 = 5;
    const VALUE_SHIFT: u32 = 16;
    let mut hash: u32 = 0;
    for &c in name {
        hash = (hash << NAME_SHIFT) ^ (hash >> (32 - NAME_SHIFT)) ^ (c as u32);
    }
    if !value.is_empty() {
        let mut v = value.to_vec();
        while v.len() % 4 != 0 {
            v.push(0);
        }
        for chunk in v.chunks_exact(4) {
            let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            hash = (hash << VALUE_SHIFT) ^ (hash >> (32 - VALUE_SHIFT)) ^ w;
        }
    }
    hash
}

/// Serialize attributes into a full external xattr block: a 32-byte header
/// (magic, refcount=1, blocks=1, zero hash, zero checksum) followed by the
/// entry array and values (`e_value_offs` measured from the block start). The
/// `h_checksum` is filled in by the caller. Returns None if they don't fit.
fn serialize_block(attrs: &[ParsedXattr], block_size: usize) -> Option<Vec<u8>> {
    let mut block = vec![0u8; block_size];
    block[0..4].copy_from_slice(&EXT4_XATTR_MAGIC.to_le_bytes());
    block[4..8].copy_from_slice(&1u32.to_le_bytes()); // h_refcount
    block[8..12].copy_from_slice(&1u32.to_le_bytes()); // h_blocks
    // h_hash (12..16), h_checksum (16..20), reserved (20..32) stay 0.

    let mut sorted: Vec<&ParsedXattr> = attrs.iter().collect();
    sorted.sort_by(|a, b| (a.name_index, &a.name).cmp(&(b.name_index, &b.name)));

    let mut entry_off = BLOCK_HDR;
    let mut value_end = block_size;

    for a in sorted {
        let elen = entry_len(a.name.len());
        // A value in a dedicated inode occupies no space in the block.
        let in_inode = a.value_inum != 0;
        let vpad = if in_inode { 0 } else { value_pad(a.value.len()) };
        if entry_off + elen + 4 > value_end.saturating_sub(vpad) {
            return None;
        }

        let e_value_offs = if in_inode || a.value.is_empty() {
            0u16
        } else {
            let voff = value_end - vpad;
            block[voff..voff + a.value.len()].copy_from_slice(&a.value);
            value_end = voff;
            voff as u16
        };

        block[entry_off] = a.name.len() as u8;
        block[entry_off + 1] = a.name_index;
        block[entry_off + 2..entry_off + 4].copy_from_slice(&e_value_offs.to_le_bytes());
        block[entry_off + 4..entry_off + 8].copy_from_slice(&a.value_inum.to_le_bytes());
        block[entry_off + 8..entry_off + 12]
            .copy_from_slice(&(a.value.len() as u32).to_le_bytes());
        // For a value in a dedicated inode the entry hash folds in the value's
        // hash word rather than the (absent) inline bytes.
        let e_hash = if in_inode {
            let vhash = hash_entry(&[], &a.value);
            let mut h = hash_entry(&a.name, &[]);
            h = (h << 16) ^ (h >> 16) ^ vhash;
            h
        } else {
            hash_entry(&a.name, &a.value)
        };
        block[entry_off + 12..entry_off + 16].copy_from_slice(&e_hash.to_le_bytes());
        block[entry_off + ENTRY_FIXED..entry_off + ENTRY_FIXED + a.name.len()]
            .copy_from_slice(&a.name);

        entry_off += elen;
    }
    Some(block)
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

/// A decoded attribute. `value_inum` is 0 for an inline value; when reading it
/// names the inode holding the value (resolved by xattr_collect), and when
/// writing it marks a value to be stored in that dedicated inode.
struct ParsedXattr {
    name_index: u8,
    name: Vec<u8>,
    value: Vec<u8>,
    value_inum: u32,
    value_size: usize,
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

        // An inline value is read here; a value in a dedicated inode
        // (e_value_inum != 0) is resolved later by xattr_collect.
        let value = if e_value_inum == 0 && e_value_offs + e_value_size <= value_base.len() {
            value_base[e_value_offs..e_value_offs + e_value_size].to_vec()
        } else {
            Vec::new()
        };

        out.push(ParsedXattr {
            name_index,
            name,
            value,
            value_inum: e_value_inum,
            value_size: e_value_size,
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

        // Resolve any value stored in a dedicated inode by reading its data,
        // then treat it as an ordinary inline value from here on.
        for attr in out.iter_mut() {
            if attr.value_inum != 0 {
                let mut buf = vec![0u8; attr.value_size];
                if self.read_at(attr.value_inum, 0, &mut buf).is_ok() {
                    attr.value = buf;
                }
                attr.value_inum = 0;
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

    /// Byte offset of the ibody xattr area within the inode record, and the
    /// record size — or None when the inode is too small to hold any.
    fn ibody_geometry(&self, inode: &Ext4Inode) -> Option<(usize, usize)> {
        let inode_size = self.super_block.inode_size() as usize;
        let extra = inode.i_extra_isize() as usize;
        let off = EXT4_GOOD_OLD_INODE_SIZE as usize + extra;
        if extra > 0 && off + 4 < inode_size {
            Some((off, inode_size))
        } else {
            None
        }
    }

    /// Parse only the in-inode attributes.
    fn read_ibody_attrs(&self, ino: u32, inode: &Ext4Inode) -> Vec<ParsedXattr> {
        let (off, inode_size) = match self.ibody_geometry(inode) {
            Some(g) => g,
            None => return Vec::new(),
        };
        let record = self
            .block_device
            .read_offset(self.inode_disk_pos(ino), inode_size);
        let magic = u32::from_le_bytes([
            record[off],
            record[off + 1],
            record[off + 2],
            record[off + 3],
        ]);
        if magic != EXT4_XATTR_MAGIC {
            return Vec::new();
        }
        let region = &record[off + 4..inode_size];
        parse_entries(region, region)
    }

    /// Write `attrs` into the inode body, refreshing the inode checksum. Returns
    /// ENOSPC if they don't fit (block spillover is handled by the caller).
    fn write_ibody_attrs(&self, ino: u32, attrs: &[ParsedXattr]) -> Result<()> {
        let mut inode_ref = self.get_inode_ref(ino);
        let (off, inode_size) = match self.ibody_geometry(&inode_ref.inode) {
            Some(g) => g,
            // No in-inode xattr space: nothing to clear; only a real store fails.
            None if attrs.is_empty() => return Ok(()),
            None => return_errno_with_message!(Errno::ENOSPC, "inode has no xattr space"),
        };
        let inode_pos = self.inode_disk_pos(ino);

        if attrs.is_empty() {
            // No attributes left: clear the header so the area reads as empty.
            let zeros = vec![0u8; inode_size - off];
            self.block_device.write_offset(inode_pos + off, &zeros);
            self.write_back_inode(&mut inode_ref);
            return Ok(());
        }

        let region_len = inode_size - off - 4;
        let region = match serialize_region(attrs, region_len) {
            Some(r) => r,
            None => return_errno_with_message!(Errno::ENOSPC, "xattrs do not fit in inode body"),
        };

        // Magic followed by the entry/value region.
        let mut bytes = Vec::with_capacity(4 + region.len());
        bytes.extend_from_slice(&EXT4_XATTR_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&region);
        self.block_device.write_offset(inode_pos + off, &bytes);

        // Recompute the inode checksum over the new record tail.
        self.write_back_inode(&mut inode_ref);
        Ok(())
    }

    /// Set (create or replace) an extended attribute. `flags` may be
    /// XATTR_CREATE (fail if present) or XATTR_REPLACE (fail if absent).
    pub fn xattr_set(&self, ino: u32, name: &str, value: &[u8], flags: i32) -> Result<()> {
        let (idx, suffix) = match split_name(name) {
            Some(v) => v,
            None => return_errno_with_message!(Errno::ENOTSUP, "unsupported xattr namespace"),
        };

        let mut attrs = self.xattr_collect(ino);
        let exists = attrs
            .iter()
            .any(|a| a.name_index == idx && a.name == suffix.as_bytes());

        if flags & XATTR_CREATE != 0 && exists {
            return_errno_with_message!(Errno::EEXIST, "xattr exists");
        }
        if flags & XATTR_REPLACE != 0 && !exists {
            return_errno_with_message!(Errno::ENODATA, "xattr does not exist");
        }

        attrs.retain(|a| !(a.name_index == idx && a.name == suffix.as_bytes()));
        attrs.push(ParsedXattr {
            name_index: idx,
            name: suffix.as_bytes().to_vec(),
            value: value.to_vec(),
            value_inum: 0,
            value_size: value.len(),
        });

        self.xattr_store(ino, attrs)
    }

    /// Remove an extended attribute, or ENODATA if it is not present.
    pub fn xattr_remove(&self, ino: u32, name: &str) -> Result<()> {
        let (idx, suffix) = match split_name(name) {
            Some(v) => v,
            None => return_errno_with_message!(Errno::ENODATA, "unsupported xattr namespace"),
        };

        let mut attrs = self.xattr_collect(ino);
        let before = attrs.len();
        attrs.retain(|a| !(a.name_index == idx && a.name == suffix.as_bytes()));
        if attrs.len() == before {
            return_errno_with_message!(Errno::ENODATA, "no such xattr");
        }

        self.xattr_store(ino, attrs)
    }

    /// Persist the full attribute set, choosing storage: the inode body if
    /// everything fits and no external block is in use, otherwise the external
    /// block. An empty set clears both and frees the block.
    fn xattr_store(&self, ino: u32, attrs: Vec<ParsedXattr>) -> Result<()> {
        let inode = self.get_inode_ref(ino).inode;
        let has_block = self.xattr_block_of(&inode) != 0;

        if attrs.is_empty() {
            self.write_ibody_attrs(ino, &[])?;
            if has_block {
                self.free_xattr_block(ino)?;
            }
            return Ok(());
        }

        let fits_ibody = !has_block
            && match self.ibody_geometry(&inode) {
                Some((off, inode_size)) => {
                    serialize_region(&attrs, inode_size - off - 4).is_some()
                }
                None => false,
            };

        if fits_ibody {
            self.write_ibody_attrs(ino, &attrs)
        } else {
            // Everything lives in the block; keep the body clear.
            self.write_ibody_attrs(ino, &[])?;
            self.write_block_attrs(ino, &attrs)
        }
    }

    /// crc32c of an external xattr block (uuid seed + block number + block with
    /// the checksum field zeroed). Zero when metadata checksums are disabled.
    fn xattr_block_checksum(&self, block_nr: u64, block: &[u8]) -> u32 {
        if self.super_block.features_read_only & 0x400 == 0 {
            return 0;
        }
        let uuid = &self.super_block.uuid;
        let seed = ext4_crc32c(EXT4_CRC32_INIT, uuid, uuid.len() as u32);
        let mut c = ext4_crc32c(seed, &block_nr.to_le_bytes(), 8);
        // Header up to h_checksum, then the zeroed checksum field, then the rest.
        c = ext4_crc32c(c, &block[0..16], 16);
        c = ext4_crc32c(c, &[0u8; 4], 4);
        c = ext4_crc32c(c, &block[20..], (block.len() - 20) as u32);
        c
    }

    /// Write the whole attribute set into the external block (allocating it on
    /// first use), and point the inode at it. Values too large to store inline
    /// are spilled into dedicated inodes when the ea_inode feature is enabled.
    fn write_block_attrs(&self, ino: u32, attrs: &[ParsedXattr]) -> Result<()> {
        // The block is rewritten wholesale, so drop the ea_inodes it referenced
        // before; large values below are stored in fresh ones.
        let old_ea_units = self.free_block_ea_inodes(ino)?;

        let bs = self.block_size();
        let ea_feature =
            self.super_block.incompat_features() & EXT4_FEATURE_INCOMPAT_EA_INODE != 0;

        let mut prepared: Vec<ParsedXattr> = Vec::with_capacity(attrs.len());
        let mut new_ea_units: u64 = 0;
        for a in attrs {
            let mut p = ParsedXattr {
                name_index: a.name_index,
                name: a.name.clone(),
                value: a.value.clone(),
                value_inum: 0,
                value_size: a.value.len(),
            };
            // A value that can't share the block goes into its own inode.
            if ea_feature && a.value.len() >= bs {
                let (inum, units) = self.create_ea_inode(ino, &a.value)?;
                p.value_inum = inum;
                new_ea_units += units;
            }
            prepared.push(p);
        }

        let mut block = match serialize_block(&prepared, bs) {
            Some(b) => b,
            None => return_errno_with_message!(Errno::ENOSPC, "xattrs do not fit in a block"),
        };

        let mut inode_ref = self.get_inode_ref(ino);
        let mut blk = self.xattr_block_of(&inode_ref.inode);
        if blk == 0 {
            blk = self.balloc_alloc_block(&mut inode_ref, None)?;
            inode_ref.inode.file_acl = blk as u32;
            inode_ref.inode.osd2.l_i_file_acl_high = (blk >> 32) as u16;
        }

        // A value inode's blocks are charged to the owning inode (quota model).
        let charged = inode_ref.inode.blocks_count() + new_ea_units - old_ea_units;
        inode_ref.inode.set_blocks_count(charged);

        let csum = self.xattr_block_checksum(blk, &block);
        block[16..20].copy_from_slice(&csum.to_le_bytes());
        self.block_device.write_offset(blk as usize * bs, &block);

        self.write_back_inode(&mut inode_ref);
        Ok(())
    }

    /// Free the external xattr block (and any value inodes it points at) and
    /// clear the inode's pointer to it.
    fn free_xattr_block(&self, ino: u32) -> Result<()> {
        let old_ea_units = self.free_block_ea_inodes(ino)?;

        let mut inode_ref = self.get_inode_ref(ino);
        let blk = self.xattr_block_of(&inode_ref.inode);
        if blk == 0 {
            return Ok(());
        }
        inode_ref.inode.file_acl = 0;
        inode_ref.inode.osd2.l_i_file_acl_high = 0;
        // Drop the value inodes' charge from the owner before writing it back.
        let charged = inode_ref.inode.blocks_count().saturating_sub(old_ea_units);
        inode_ref.inode.set_blocks_count(charged);
        self.write_back_inode(&mut inode_ref);
        self.balloc_free_blocks(&mut inode_ref, blk, 1);
        Ok(())
    }

    /// Free every value inode referenced by the current external block,
    /// returning the total i_blocks (512-byte units) they held.
    fn free_block_ea_inodes(&self, ino: u32) -> Result<u64> {
        let inode = self.get_inode_ref(ino).inode;
        let blk = self.xattr_block_of(&inode);
        if blk == 0 {
            return Ok(0);
        }
        let bs = self.block_size();
        let data = self.block_device.read_offset(blk as usize * bs, bs);
        if u32::from_le_bytes([data[0], data[1], data[2], data[3]]) != EXT4_XATTR_MAGIC {
            return Ok(0);
        }
        let mut units = 0u64;
        for e in parse_entries(&data[BLOCK_HDR..], &data) {
            if e.value_inum != 0 {
                units += self.get_inode_ref(e.value_inum).inode.blocks_count();
                self.free_ea_inode(e.value_inum)?;
            }
        }
        Ok(units)
    }

    /// Create a dedicated inode holding `value`, owned (via the back-reference)
    /// by `parent`. Returns its inode number and its i_blocks (512-byte units).
    fn create_ea_inode(&self, parent: u32, value: &[u8]) -> Result<(u32, u64)> {
        let ino = self.ialloc_alloc_inode(false)?;
        let value_hash = hash_entry(&[], value);

        let mut iref = self.get_inode_ref(ino);
        iref.inode = Ext4Inode::default();
        iref.inode.set_mode(InodeFileType::S_IFREG.bits() | 0o600);
        iref.inode.set_links_count(1);
        if self.super_block.inode_size() > EXT4_GOOD_OLD_INODE_SIZE {
            iref.inode.set_i_extra_isize(self.super_block.extra_size());
        }
        iref.inode
            .set_flags(EXT4_INODE_FLAG_EXTENTS as u32 | EXT4_EA_INODE_FL);
        iref.inode.extent_tree_init();
        self.write_back_inode_without_csum(&iref);

        // Store the value as the inode's data.
        self.write_at(ino, 0, value)?;

        // write_at reloaded and rewrote the inode; set the ea_inode bookkeeping
        // e2fsck validates: the value hash (i_atime), the reference count
        // (i_ctime high : osd1 low, here 1), and the owner back-ref (i_mtime).
        let mut iref = self.get_inode_ref(ino);
        iref.inode.atime = value_hash;
        iref.inode.ctime = 0;
        iref.inode.osd1 = 1;
        iref.inode.mtime = parent;
        iref.inode.set_links_count(1);
        let flags = iref.inode.flags() | EXT4_EA_INODE_FL;
        iref.inode.set_flags(flags);
        self.write_back_inode(&mut iref);

        let units = iref.inode.blocks_count();
        Ok((ino, units))
    }

    /// Free a value inode and its data.
    fn free_ea_inode(&self, inum: u32) -> Result<()> {
        let mut iref = self.get_inode_ref(inum);
        self.truncate_inode(&mut iref, 0)?;
        iref.inode = Ext4Inode::default();
        self.write_back_inode_without_csum(&iref);
        self.ialloc_free_inode(inum, false);
        Ok(())
    }
}
