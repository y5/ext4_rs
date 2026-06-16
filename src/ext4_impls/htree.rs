//! ext4 HTree (hashed B-tree) directory indexing.
//!
//! This module ports the directory hash from `fs/ext4/hash.c` and (in later
//! phases) the dx_root/dx_node index traversal and maintenance.

use crate::ext4_defs::*;
use crate::prelude::*;
use crate::return_errno_with_message;

/// Byte offset of the `dx_entry` array within a dx_root block: fake `.` (12) +
/// fake `..` (12) + `dx_root_info` (8).
const DX_ROOT_ENTRIES_OFFSET: usize = 0x20;
/// Byte offset of the `dx_entry` array within a dx_node block: an 8-byte fake
/// dirent claiming the whole block.
const DX_NODE_ENTRIES_OFFSET: usize = 8;

/// Directory hash versions (`s_def_hash_version` / `dx_root_info.hash_version`).
pub const DX_HASH_LEGACY: u8 = 0;
pub const DX_HASH_HALF_MD4: u8 = 1;
pub const DX_HASH_TEA: u8 = 2;
pub const DX_HASH_LEGACY_UNSIGNED: u8 = 3;
pub const DX_HASH_HALF_MD4_UNSIGNED: u8 = 4;
pub const DX_HASH_TEA_UNSIGNED: u8 = 5;

/// `EXT4_HTREE_EOF_32BIT` — the reserved "end of htree" hash value (pre-shift).
const EXT4_HTREE_EOF_32BIT: u32 = 0x7fff_ffff;

const TEA_DELTA: u32 = 0x9E37_79B9;
// half_md4 round constants (0, sqrt(2)*2^30, sqrt(3)*2^30).
const K1: u32 = 0;
const K2: u32 = 0x5A82_7999;
const K3: u32 = 0x6ED9_EBA1;

#[inline]
fn rol32(x: u32, s: u32) -> u32 {
    x.rotate_left(s)
}

/// `half_md4_transform` from `lib/halfmd4.c`. Mutates `buf`, returns `buf[1]`.
fn half_md4_transform(buf: &mut [u32; 4], inb: &[u32; 8]) -> u32 {
    let (mut a, mut b, mut c, mut d) = (buf[0], buf[1], buf[2], buf[3]);

    // F(x,y,z) = z ^ (x & (y ^ z))
    macro_rules! round_f {
        ($a:ident,$b:ident,$c:ident,$d:ident,$x:expr,$s:expr) => {
            let f = $d ^ ($b & ($c ^ $d));
            $a = rol32($a.wrapping_add(f).wrapping_add($x), $s);
        };
    }
    // G(x,y,z) = (x & y) + ((x ^ y) & z)
    macro_rules! round_g {
        ($a:ident,$b:ident,$c:ident,$d:ident,$x:expr,$s:expr) => {
            let g = ($b & $c).wrapping_add(($b ^ $c) & $d);
            $a = rol32($a.wrapping_add(g).wrapping_add($x), $s);
        };
    }
    // H(x,y,z) = x ^ y ^ z
    macro_rules! round_h {
        ($a:ident,$b:ident,$c:ident,$d:ident,$x:expr,$s:expr) => {
            let h = $b ^ $c ^ $d;
            $a = rol32($a.wrapping_add(h).wrapping_add($x), $s);
        };
    }

    // Round 1
    round_f!(a, b, c, d, inb[0].wrapping_add(K1), 3);
    round_f!(d, a, b, c, inb[1].wrapping_add(K1), 7);
    round_f!(c, d, a, b, inb[2].wrapping_add(K1), 11);
    round_f!(b, c, d, a, inb[3].wrapping_add(K1), 19);
    round_f!(a, b, c, d, inb[4].wrapping_add(K1), 3);
    round_f!(d, a, b, c, inb[5].wrapping_add(K1), 7);
    round_f!(c, d, a, b, inb[6].wrapping_add(K1), 11);
    round_f!(b, c, d, a, inb[7].wrapping_add(K1), 19);

    // Round 2
    round_g!(a, b, c, d, inb[1].wrapping_add(K2), 3);
    round_g!(d, a, b, c, inb[3].wrapping_add(K2), 5);
    round_g!(c, d, a, b, inb[5].wrapping_add(K2), 9);
    round_g!(b, c, d, a, inb[7].wrapping_add(K2), 13);
    round_g!(a, b, c, d, inb[0].wrapping_add(K2), 3);
    round_g!(d, a, b, c, inb[2].wrapping_add(K2), 5);
    round_g!(c, d, a, b, inb[4].wrapping_add(K2), 9);
    round_g!(b, c, d, a, inb[6].wrapping_add(K2), 13);

    // Round 3
    round_h!(a, b, c, d, inb[3].wrapping_add(K3), 3);
    round_h!(d, a, b, c, inb[7].wrapping_add(K3), 9);
    round_h!(c, d, a, b, inb[2].wrapping_add(K3), 11);
    round_h!(b, c, d, a, inb[6].wrapping_add(K3), 15);
    round_h!(a, b, c, d, inb[1].wrapping_add(K3), 3);
    round_h!(d, a, b, c, inb[5].wrapping_add(K3), 9);
    round_h!(c, d, a, b, inb[0].wrapping_add(K3), 11);
    round_h!(b, c, d, a, inb[4].wrapping_add(K3), 15);

    buf[0] = buf[0].wrapping_add(a);
    buf[1] = buf[1].wrapping_add(b);
    buf[2] = buf[2].wrapping_add(c);
    buf[3] = buf[3].wrapping_add(d);
    buf[1]
}

/// `TEA_transform` from `fs/ext4/hash.c`. Mutates `buf[0..2]`.
fn tea_transform(buf: &mut [u32; 4], inb: &[u32; 4]) {
    let mut sum = 0u32;
    let (mut b0, mut b1) = (buf[0], buf[1]);
    let (a, b, c, d) = (inb[0], inb[1], inb[2], inb[3]);

    for _ in 0..16 {
        sum = sum.wrapping_add(TEA_DELTA);
        b0 = b0.wrapping_add(
            ((b1 << 4).wrapping_add(a)) ^ b1.wrapping_add(sum) ^ ((b1 >> 5).wrapping_add(b)),
        );
        b1 = b1.wrapping_add(
            ((b0 << 4).wrapping_add(c)) ^ b0.wrapping_add(sum) ^ ((b0 >> 5).wrapping_add(d)),
        );
    }

    buf[0] = buf[0].wrapping_add(b0);
    buf[1] = buf[1].wrapping_add(b1);
}

/// The old legacy hash (`dx_hack_hash`). `signed` selects the char signedness.
fn dx_hack_hash(name: &[u8], signed: bool) -> u32 {
    let mut hash0: u32 = 0x12a3_fe2d;
    let mut hash1: u32 = 0x37ab_e8f9;
    for &ch in name {
        // C casts the char to `int` first (sign-extending for signed char).
        let c = if signed {
            (ch as i8) as i32
        } else {
            ch as i32
        };
        let mul = (c.wrapping_mul(7152373)) as u32;
        let mut hash = hash1.wrapping_add(hash0 ^ mul);
        if hash & 0x8000_0000 != 0 {
            hash = hash.wrapping_sub(0x7fff_ffff);
        }
        hash1 = hash0;
        hash0 = hash;
    }
    hash0 << 1
}

/// `str2hashbuf` — pack `num` u32 words from `msg`, padding by length.
fn str2hashbuf(msg: &[u8], num: usize, signed: bool) -> [u32; 8] {
    let len = msg.len();
    let mut pad = (len as u32) | ((len as u32) << 8);
    pad |= pad << 16;

    let mut buf = [pad; 8];
    let mut val = pad;
    let mut written = 0usize;
    let mut remaining = num;

    let effective = core::cmp::min(len, num * 4);
    for i in 0..effective {
        let c = if signed {
            (msg[i] as i8) as i32
        } else {
            msg[i] as i32
        };
        val = (c as u32).wrapping_add(val << 8);
        if i % 4 == 3 {
            buf[written] = val;
            written += 1;
            val = pad;
            remaining -= 1;
        }
    }
    // The trailing `if (--num >= 0) *buf++ = val;` then pads the rest.
    if remaining > 0 {
        buf[written] = val;
        written += 1;
        remaining -= 1;
    }
    while remaining > 0 {
        buf[written] = pad;
        written += 1;
        remaining -= 1;
    }
    buf
}

/// Compute the ext4 directory hash of `name` under `hash_version` and `seed`
/// (a zero seed selects the kernel default constants). Returns
/// `(major_hash, minor_hash)`; the legacy hash has no minor hash (0).
///
/// Mirrors `ext4fs_dirhash` in `fs/ext4/hash.c`.
pub fn ext4_dir_hash(name: &[u8], hash_version: u8, seed: [u32; 4]) -> (u32, u32) {
    // Default seed, overridden only if the provided seed is non-zero.
    let mut buf: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    if seed.iter().any(|&s| s != 0) {
        buf = seed;
    }

    let mut minor_hash = 0u32;
    let major_hash;

    match hash_version {
        DX_HASH_LEGACY | DX_HASH_LEGACY_UNSIGNED => {
            let signed = hash_version == DX_HASH_LEGACY;
            major_hash = dx_hack_hash(name, signed);
        }
        DX_HASH_HALF_MD4 | DX_HASH_HALF_MD4_UNSIGNED => {
            let signed = hash_version == DX_HASH_HALF_MD4;
            let mut p = name;
            loop {
                let inb = str2hashbuf(p, 8, signed);
                half_md4_transform(&mut buf, &inb);
                if p.len() <= 32 {
                    break;
                }
                p = &p[32..];
            }
            major_hash = buf[1];
            minor_hash = buf[2];
        }
        DX_HASH_TEA | DX_HASH_TEA_UNSIGNED => {
            let signed = hash_version == DX_HASH_TEA;
            let mut p = name;
            loop {
                let full = str2hashbuf(p, 4, signed);
                let inb = [full[0], full[1], full[2], full[3]];
                tea_transform(&mut buf, &inb);
                if p.len() <= 16 {
                    break;
                }
                p = &p[16..];
            }
            major_hash = buf[0];
            minor_hash = buf[1];
        }
        _ => return (0, 0),
    }

    // Reserve bit 0, and avoid colliding with the reserved EOF hash.
    let mut hash = major_hash & !1;
    if hash == (EXT4_HTREE_EOF_32BIT << 1) {
        hash = (EXT4_HTREE_EOF_32BIT - 1) << 1;
    }
    (hash, minor_hash)
}

/// One parsed dx index block: where its entry array starts, plus its count.
struct DxBlock {
    block: Block,
    entries_off: usize,
    count: u16,
}

impl DxBlock {
    /// `(hash, child_block)` of entry `i`. Entry 0's hash slot holds the
    /// count/limit header, so its hash is meaningless — only `.1` is used.
    fn entry(&self, i: usize) -> (u32, u32) {
        let o = self.entries_off + i * 8;
        let h = u32::from_le_bytes(self.block.data[o..o + 4].try_into().unwrap());
        let b = u32::from_le_bytes(self.block.data[o + 4..o + 8].try_into().unwrap());
        (h, b)
    }

    /// The child block whose hash range covers `target` (largest entry whose
    /// hash ≤ target; entry 0 covers the lowest range).
    fn lookup(&self, target: u32) -> u32 {
        let mut child = self.entry(0).1;
        let mut i = 1usize;
        while i < self.count as usize {
            let (h, b) = self.entry(i);
            if target < h {
                break;
            }
            child = b;
            i += 1;
        }
        child
    }
}

impl Ext4 {
    /// Load and sanity-check one dx index block at logical block `lblock`.
    fn dx_load_block(
        &self,
        dir: &Ext4InodeRef,
        lblock: u32,
        entries_off: usize,
    ) -> Result<DxBlock> {
        let bs = self.block_size();
        let pblock = self.get_pblock_idx(dir, lblock)?;
        let block = Block::load(&self.block_device, pblock as usize * bs, bs);
        let count = u16::from_le_bytes([block.data[entries_off + 2], block.data[entries_off + 3]]);
        let limit = u16::from_le_bytes([block.data[entries_off], block.data[entries_off + 1]]);
        if count == 0 || count > limit {
            return_errno_with_message!(Errno::EIO, "dx block: bad count/limit");
        }
        Ok(DxBlock {
            block,
            entries_off,
            count,
        })
    }

    /// Descend the htree index to the leaf logical block that would hold `name`.
    pub fn dx_probe(&self, dir: &Ext4InodeRef, name: &str) -> Result<u32> {
        // Root is logical block 0.
        let root = self.dx_load_block(dir, 0, DX_ROOT_ENTRIES_OFFSET)?;
        let info_length = root.block.data[0x1d];
        let mut indirect_levels = root.block.data[0x1e];
        if info_length != 8 {
            return_errno_with_message!(Errno::ENOTSUP, "dx_root: unexpected info_length");
        }
        let hash_version = root.block.data[0x1c];

        let seed = self.super_block.hash_seed();
        let (hash, _minor) = ext4_dir_hash(name.as_bytes(), hash_version, seed);

        let mut node = root;
        loop {
            let child = node.lookup(hash);
            if indirect_levels == 0 {
                return Ok(child); // leaf logical block
            }
            node = self.dx_load_block(dir, child, DX_NODE_ENTRIES_OFFSET)?;
            indirect_levels -= 1;
        }
    }

    /// Look up `name` in an htree-indexed directory via the index. Populates
    /// `result` like `dir_find_entry`. Errors with `ENOTSUP` if the directory
    /// is not indexed, `ENOENT` if the name is absent from its hash leaf.
    pub fn dx_find_entry(
        &self,
        parent_inode: u32,
        name: &str,
        result: &mut Ext4DirSearchResult,
    ) -> Result<usize> {
        let dir = self.get_inode_ref(parent_inode);
        if !dir.inode.is_index() {
            return_errno_with_message!(Errno::ENOTSUP, "dx_find_entry on non-indexed directory");
        }
        let leaf = self.dx_probe(&dir, name)?;
        let bs = self.block_size();
        let pblock = self.get_pblock_idx(&dir, leaf)?;
        let block = Block::load(&self.block_device, pblock as usize * bs, bs);
        self.dir_find_in_block(&block, name, result)?;
        result.pblock_id = pblock as usize;
        Ok(EOK)
    }
}

// --- HTree index maintenance (create / insert / split) ---------------------

use crate::utils::{ext4_crc32c, EXT4_CRC32_INIT};

const DX_ENTRY_SIZE: usize = 8;
const DX_TAIL_SIZE: usize = 8;

#[inline]
fn dx_count(d: &[u8], eo: usize) -> u16 {
    u16::from_le_bytes([d[eo + 2], d[eo + 3]])
}
#[inline]
fn dx_limit(d: &[u8], eo: usize) -> u16 {
    u16::from_le_bytes([d[eo], d[eo + 1]])
}
#[inline]
fn dx_set_count(d: &mut [u8], eo: usize, c: u16) {
    d[eo + 2..eo + 4].copy_from_slice(&c.to_le_bytes());
}
#[inline]
fn dx_set_limit(d: &mut [u8], eo: usize, l: u16) {
    d[eo..eo + 2].copy_from_slice(&l.to_le_bytes());
}
#[inline]
fn dx_get_entry(d: &[u8], eo: usize, i: usize) -> (u32, u32) {
    let o = eo + i * DX_ENTRY_SIZE;
    (
        u32::from_le_bytes(d[o..o + 4].try_into().unwrap()),
        u32::from_le_bytes(d[o + 4..o + 8].try_into().unwrap()),
    )
}
#[inline]
fn dx_set_entry(d: &mut [u8], eo: usize, i: usize, hash: u32, blk: u32) {
    let o = eo + i * DX_ENTRY_SIZE;
    d[o..o + 4].copy_from_slice(&hash.to_le_bytes());
    d[o + 4..o + 8].copy_from_slice(&blk.to_le_bytes());
}

/// Aligned on-disk length of a directory entry with a `name_len`-byte name.
fn aligned_de_len(name_len: usize) -> usize {
    let mut l = size_of::<Ext4FakeDirEntry>() + name_len;
    if l % 4 != 0 {
        l += 4 - (l % 4);
    }
    l
}

impl Ext4 {
    /// Max dx entries that fit in a block whose entry array starts at `eo`,
    /// reserving room for the trailing 8-byte dx_tail (metadata_csum).
    fn dx_block_limit(&self, eo: usize) -> u16 {
        ((self.block_size() - eo - DX_TAIL_SIZE) / DX_ENTRY_SIZE) as u16
    }

    /// Compute and store a dx index block's checksum in its trailing dx_tail.
    fn dx_set_csum(&self, block: &mut Block, dir_ino: u32, ino_gen: u32, eo: usize) {
        let count = dx_count(&block.data, eo) as usize;
        let limit = dx_limit(&block.data, eo) as usize;
        let tail_off = eo + limit * DX_ENTRY_SIZE;

        // Inode csum seed: crc32c(crc32c(crc32c(INIT, uuid), ino), generation).
        let uuid = self.super_block.uuid;
        let mut seed = ext4_crc32c(EXT4_CRC32_INIT, &uuid, uuid.len() as u32);
        seed = ext4_crc32c(seed, &dir_ino.to_le_bytes(), 4);
        seed = ext4_crc32c(seed, &ino_gen.to_le_bytes(), 4);

        // crc over the header+entries, then the dx_tail with a zeroed checksum
        // (dt_reserved + dt_checksum = 8 zero bytes).
        let size = eo + count * DX_ENTRY_SIZE;
        let mut csum = ext4_crc32c(seed, &block.data[..size], size as u32);
        csum = ext4_crc32c(csum, &[0u8; DX_TAIL_SIZE], DX_TAIL_SIZE as u32);

        block.data[tail_off..tail_off + 4].copy_from_slice(&0u32.to_le_bytes());
        block.data[tail_off + 4..tail_off + 8].copy_from_slice(&csum.to_le_bytes());
    }

    /// The directory's hash version (from its dx_root) and the fs hash seed.
    fn dx_hash_params(&self, dir: &Ext4InodeRef) -> Result<(u8, [u32; 4])> {
        let bs = self.block_size();
        let p0 = self.get_pblock_idx(dir, 0)?;
        let root = Block::load(&self.block_device, p0 as usize * bs, bs);
        Ok((root.data[0x1c], self.super_block.hash_seed()))
    }

    /// Walk a linear leaf block, returning each used entry as (name, inode,
    /// file_type). Skips `.`/`..` and unused slots.
    fn dx_read_leaf_entries(&self, data: &[u8]) -> Vec<(String, u32, u8)> {
        let mut out = Vec::new();
        let limit = self.block_size() - size_of::<Ext4DirEntryTail>();
        let mut off = 0;
        while off < limit {
            let de = match Ext4DirEntry::try_from(&data[off..]) {
                Ok(de) => de,
                Err(_) => break,
            };
            let rl = de.entry_len() as usize;
            if rl == 0 {
                break;
            }
            if !de.unused() && de.name_len > 0 {
                let name = de.get_name();
                if name != "." && name != ".." {
                    out.push((name, de.inode, de.get_de_type()));
                }
            }
            off += rl;
        }
        out
    }

    /// Write `items` into `block` as a fresh linear leaf (last entry spans to
    /// the tail), then set the tail checksum.
    fn dx_write_leaf(
        &self,
        block: &mut Block,
        items: &[(String, u32, u8)],
        dir_ino: u32,
        ino_gen: u32,
    ) {
        for b in block.data.iter_mut() {
            *b = 0;
        }
        let tail_start = self.block_size() - size_of::<Ext4DirEntryTail>();

        if items.is_empty() {
            let mut de = Ext4DirEntry::default();
            de.write_entry(tail_start as u16, 0, "", DirEntryType::EXT4_DE_UNKNOWN);
            de.copy_to_slice(&mut block.data, 0);
        } else {
            let mut off = 0;
            for (i, (name, ino, ty)) in items.iter().enumerate() {
                let minlen = aligned_de_len(name.len());
                let rec = if i == items.len() - 1 {
                    tail_start - off
                } else {
                    minlen
                };
                let mut de = Ext4DirEntry::default();
                de.write_entry(rec as u16, *ino, name, DirEntryType::from_bits_truncate(*ty));
                de.copy_to_slice(&mut block.data, off);
                off += minlen;
            }
        }

        let tail = Ext4DirEntryTail::new();
        tail.copy_to_slice(&mut block.data);
        self.dir_set_csum(block, dir_ino, ino_gen);
    }

    /// Format `block` as a dx_root: fake `.`/`..`, dx_root_info, and a single
    /// dx_entry covering the whole hash range and pointing at `first_leaf`.
    fn dx_format_root(
        &self,
        block: &mut Block,
        dir_ino: u32,
        dotdot_ino: u32,
        hash_version: u8,
        first_leaf: u32,
    ) {
        let bs = self.block_size();
        for b in block.data.iter_mut() {
            *b = 0;
        }

        let mut dot = Ext4DirEntry::default();
        dot.write_entry(12, dir_ino, ".", DirEntryType::EXT4_DE_DIR);
        dot.copy_to_slice(&mut block.data, 0);

        let mut dotdot = Ext4DirEntry::default();
        dotdot.write_entry((bs - 12) as u16, dotdot_ino, "..", DirEntryType::EXT4_DE_DIR);
        dotdot.copy_to_slice(&mut block.data, 12);

        // dx_root_info @ 0x18: reserved(4)=0, hash_version, info_length=8, levels=0
        block.data[0x1c] = hash_version;
        block.data[0x1d] = 8;
        block.data[0x1e] = 0;
        block.data[0x1f] = 0;

        let eo = DX_ROOT_ENTRIES_OFFSET;
        dx_set_limit(&mut block.data, eo, self.dx_block_limit(eo));
        dx_set_count(&mut block.data, eo, 1);
        // entry[0]'s hash slot is the count/limit header; only its block matters.
        block.data[eo + 4..eo + 8].copy_from_slice(&first_leaf.to_le_bytes());
        // The caller sets the dx_tail checksum once the block is fully built.
    }

    /// Convert a full single-block directory into an HTree: move its entries to
    /// a new leaf (logical block 1) and reformat block 0 as the dx_root.
    pub(crate) fn dx_convert_to_htree(&self, parent: &mut Ext4InodeRef) -> Result<()> {
        let bs = self.block_size();
        let gen = parent.inode.generation();
        let dir_ino = parent.inode_num;

        let p0 = self.get_pblock_idx(parent, 0)?;
        let block0 = Block::load(&self.block_device, p0 as usize * bs, bs);

        // Capture `..` (grandparent inode) and all real entries.
        let mut dotdot = 0u32;
        {
            let limit = bs - size_of::<Ext4DirEntryTail>();
            let mut off = 0;
            while off < limit {
                let de = match Ext4DirEntry::try_from(&block0.data[off..]) {
                    Ok(de) => de,
                    Err(_) => break,
                };
                let rl = de.entry_len() as usize;
                if rl == 0 {
                    break;
                }
                if !de.unused() && de.get_name() == ".." {
                    dotdot = de.inode;
                }
                off += rl;
            }
        }
        let items = self.dx_read_leaf_entries(&block0.data);

        // Allocate leaf block 1 and move the entries there.
        let leaf_lblk = (parent.inode.size() / bs as u64) as u32;
        let leaf_pblk = self.append_inode_pblk(parent)?;
        let mut leaf = Block::load(&self.block_device, leaf_pblk as usize * bs, bs);
        self.dx_write_leaf(&mut leaf, &items, dir_ino, gen);
        leaf.sync_blk_to_disk(&self.block_device);

        // Reformat block 0 as the dx_root pointing at the new leaf.
        let hash_version = self.super_block.default_hash_version();
        let mut root = Block::load(&self.block_device, p0 as usize * bs, bs);
        self.dx_format_root(&mut root, dir_ino, dotdot, hash_version, leaf_lblk);
        // Recompute the root csum with the real generation.
        self.dx_set_csum(&mut root, dir_ino, gen, DX_ROOT_ENTRIES_OFFSET);
        root.sync_blk_to_disk(&self.block_device);

        // Mark the directory indexed.
        let flags = parent.inode.flags() | (EXT4_INODE_FLAG_INDEX as u32);
        parent.inode.set_flags(flags);
        self.write_back_inode(parent);
        Ok(())
    }

    /// Write a dx_node block: an 8-byte fake dirent header claiming the whole
    /// block, then the `ents` (hash → block) array (entry 0's hash slot is the
    /// count/limit header, so only its block is kept), then the dx_tail csum.
    fn dx_write_node(&self, block: &mut Block, ents: &[(u32, u32)], dir_ino: u32, ino_gen: u32) {
        let bs = self.block_size();
        let eo = DX_NODE_ENTRIES_OFFSET;
        for b in block.data.iter_mut() {
            *b = 0;
        }
        // Fake dirent (inode 0, name_len 0) whose rec_len claims the whole block.
        block.data[4..6].copy_from_slice(&(bs as u16).to_le_bytes());
        dx_set_limit(&mut block.data, eo, self.dx_block_limit(eo));
        dx_set_count(&mut block.data, eo, ents.len() as u16);
        for (i, (h, b)) in ents.iter().enumerate() {
            if i == 0 {
                block.data[eo + 4..eo + 8].copy_from_slice(&b.to_le_bytes());
            } else {
                dx_set_entry(&mut block.data, eo, i, *h, *b);
            }
        }
        self.dx_set_csum(block, dir_ino, ino_gen, eo);
    }

    /// Hash `name` with the directory's hash version + the fs seed.
    fn dx_name_hash(&self, dir: &Ext4InodeRef, name: &str) -> Result<u32> {
        let (hv, seed) = self.dx_hash_params(dir)?;
        Ok(ext4_dir_hash(name.as_bytes(), hv, seed).0)
    }

    /// `(lblock, entries_offset)` of every index block from the root down to
    /// the leaf's parent, plus the target leaf's logical block, for `hash`.
    fn dx_probe_frames_hash(
        &self,
        dir: &Ext4InodeRef,
        hash: u32,
    ) -> Result<(Vec<(u32, usize)>, u32)> {
        let bs = self.block_size();
        let p0 = self.get_pblock_idx(dir, 0)?;
        let mut block = Block::load(&self.block_device, p0 as usize * bs, bs);
        let mut levels = block.data[0x1e];

        let mut frames = Vec::new();
        let mut lblk = 0u32;
        let mut eo = DX_ROOT_ENTRIES_OFFSET;
        loop {
            let count = dx_count(&block.data, eo);
            // Largest entry whose hash ≤ target (entry 0 covers the low range).
            let mut child = dx_get_entry(&block.data, eo, 0).1;
            let mut i = 1usize;
            while i < count as usize {
                let (h, b) = dx_get_entry(&block.data, eo, i);
                if hash < h {
                    break;
                }
                child = b;
                i += 1;
            }
            frames.push((lblk, eo));
            if levels == 0 {
                return Ok((frames, child));
            }
            let cp = self.get_pblock_idx(dir, child)?;
            block = Block::load(&self.block_device, cp as usize * bs, bs);
            lblk = child;
            eo = DX_NODE_ENTRIES_OFFSET;
            levels -= 1;
        }
    }

    /// Insert `(hash → blk)` into the index block at `frames[level]`, splitting
    /// the node — or growing the tree's depth at the root — when it is full.
    fn dx_add_to_index(
        &self,
        parent: &mut Ext4InodeRef,
        frames: &[(u32, usize)],
        level: usize,
        hash: u32,
        blk: u32,
    ) -> Result<()> {
        let bs = self.block_size();
        let (lblk, eo) = frames[level];
        let pblk = self.get_pblock_idx(parent, lblk)?;
        let mut block = Block::load(&self.block_device, pblk as usize * bs, bs);
        let count = dx_count(&block.data, eo) as usize;
        let limit = dx_limit(&block.data, eo) as usize;

        if count < limit {
            let mut pos = count;
            for i in 1..count {
                if dx_get_entry(&block.data, eo, i).0 > hash {
                    pos = i;
                    break;
                }
            }
            for i in (pos..count).rev() {
                let (h, b) = dx_get_entry(&block.data, eo, i);
                dx_set_entry(&mut block.data, eo, i + 1, h, b);
            }
            dx_set_entry(&mut block.data, eo, pos, hash, blk);
            dx_set_count(&mut block.data, eo, (count + 1) as u16);
            self.dx_set_csum(&mut block, parent.inode_num, parent.inode.generation(), eo);
            block.sync_blk_to_disk(&self.block_device);
            return Ok(());
        }

        if level == 0 {
            // Root full: add a level, then retry the insert into the new node.
            self.dx_grow_depth(parent)?;
            let (new_frames, _leaf) = self.dx_probe_frames_hash(parent, hash)?;
            let last = new_frames.len() - 1;
            return self.dx_add_to_index(parent, &new_frames, last, hash, blk);
        }

        // Split this node and register the median one level up.
        let (median, new_node) = self.dx_split_node(parent, lblk, hash, blk)?;
        self.dx_add_to_index(parent, frames, level - 1, median, new_node)
    }

    /// Split the full dx_node at `node_lblk` (after inserting `ins_*`), moving
    /// its upper half to a new node. Returns `(median_hash, new_node_lblock)`.
    fn dx_split_node(
        &self,
        parent: &mut Ext4InodeRef,
        node_lblk: u32,
        ins_hash: u32,
        ins_blk: u32,
    ) -> Result<(u32, u32)> {
        let bs = self.block_size();
        let gen = parent.inode.generation();
        let dir_ino = parent.inode_num;
        let eo = DX_NODE_ENTRIES_OFFSET;

        let node_pblk = self.get_pblock_idx(parent, node_lblk)?;
        let nb = Block::load(&self.block_device, node_pblk as usize * bs, bs);
        let count = dx_count(&nb.data, eo) as usize;
        let mut ents: Vec<(u32, u32)> = (0..count)
            .map(|i| {
                let (h, b) = dx_get_entry(&nb.data, eo, i);
                (if i == 0 { 0 } else { h }, b)
            })
            .collect();
        let mut pos = ents.len();
        for i in 1..ents.len() {
            if ents[i].0 > ins_hash {
                pos = i;
                break;
            }
        }
        ents.insert(pos, (ins_hash, ins_blk));

        let total = ents.len();
        let mut mid = total / 2;
        while mid < total && ents[mid].0 == ents[mid - 1].0 {
            mid += 1;
        }
        if mid >= total {
            mid = total / 2;
            while mid > 0 && ents[mid].0 == ents[mid - 1].0 {
                mid -= 1;
            }
        }
        if mid == 0 || mid >= total {
            return_errno_with_message!(Errno::ENOSPC, "dx node dominated by one hash");
        }
        let median = ents[mid].0;

        let new_lblk = (parent.inode.size() / bs as u64) as u32;
        let new_pblk = self.append_inode_pblk(parent)?;

        let mut oldb = Block::load(&self.block_device, node_pblk as usize * bs, bs);
        self.dx_write_node(&mut oldb, &ents[..mid], dir_ino, gen);
        oldb.sync_blk_to_disk(&self.block_device);

        let mut newb = Block::load(&self.block_device, new_pblk as usize * bs, bs);
        self.dx_write_node(&mut newb, &ents[mid..], dir_ino, gen);
        newb.sync_blk_to_disk(&self.block_device);

        Ok((median, new_lblk))
    }

    /// Grow the tree by one level: move all dx_root entries into a fresh
    /// dx_node, leave the root pointing only at it, and bump `indirect_levels`.
    fn dx_grow_depth(&self, parent: &mut Ext4InodeRef) -> Result<()> {
        let bs = self.block_size();
        let gen = parent.inode.generation();
        let dir_ino = parent.inode_num;
        let eo = DX_ROOT_ENTRIES_OFFSET;

        let p0 = self.get_pblock_idx(parent, 0)?;
        let root = Block::load(&self.block_device, p0 as usize * bs, bs);
        let count = dx_count(&root.data, eo) as usize;
        let levels = root.data[0x1e];
        let ents: Vec<(u32, u32)> = (0..count)
            .map(|i| {
                let (h, b) = dx_get_entry(&root.data, eo, i);
                (if i == 0 { 0 } else { h }, b)
            })
            .collect();

        let node_lblk = (parent.inode.size() / bs as u64) as u32;
        let node_pblk = self.append_inode_pblk(parent)?;
        let mut node = Block::load(&self.block_device, node_pblk as usize * bs, bs);
        self.dx_write_node(&mut node, &ents, dir_ino, gen);
        node.sync_blk_to_disk(&self.block_device);

        let mut root2 = Block::load(&self.block_device, p0 as usize * bs, bs);
        dx_set_count(&mut root2.data, eo, 1);
        root2.data[eo + 4..eo + 8].copy_from_slice(&node_lblk.to_le_bytes());
        root2.data[0x1e] = levels + 1;
        self.dx_set_csum(&mut root2, dir_ino, gen, eo);
        root2.sync_blk_to_disk(&self.block_device);
        Ok(())
    }

    /// Split the full leaf at `leaf_lblk` by hash, moving the upper half to a
    /// new leaf. Returns `(split_hash, new_leaf_lblock)` for the caller to
    /// register in the parent index block.
    fn dx_split_leaf(&self, parent: &mut Ext4InodeRef, leaf_lblk: u32) -> Result<(u32, u32)> {
        let bs = self.block_size();
        let gen = parent.inode.generation();
        let dir_ino = parent.inode_num;
        let (hv, seed) = self.dx_hash_params(parent)?;

        let leaf_pblk = self.get_pblock_idx(parent, leaf_lblk)?;
        let leaf = Block::load(&self.block_device, leaf_pblk as usize * bs, bs);
        let mut items: Vec<(u32, String, u32, u8)> = self
            .dx_read_leaf_entries(&leaf.data)
            .into_iter()
            .map(|(name, ino, ty)| {
                let (h, _) = ext4_dir_hash(name.as_bytes(), hv, seed);
                (h, name, ino, ty)
            })
            .collect();
        if items.len() < 2 {
            return_errno_with_message!(Errno::ENOSPC, "leaf too small to split");
        }
        items.sort_by(|a, b| a.0.cmp(&b.0));

        // Split on a hash boundary: entries sharing a hash must stay together.
        let n = items.len();
        let mut mid = n / 2;
        while mid < n && items[mid].0 == items[mid - 1].0 {
            mid += 1;
        }
        if mid >= n {
            mid = n / 2;
            while mid > 0 && items[mid].0 == items[mid - 1].0 {
                mid -= 1;
            }
        }
        if mid == 0 || mid >= n {
            return_errno_with_message!(Errno::ENOSPC, "leaf dominated by one hash");
        }
        let split_hash = items[mid].0;

        let new_lblk = (parent.inode.size() / bs as u64) as u32;
        let new_pblk = self.append_inode_pblk(parent)?;

        let lower: Vec<(String, u32, u8)> =
            items[..mid].iter().map(|x| (x.1.clone(), x.2, x.3)).collect();
        let upper: Vec<(String, u32, u8)> =
            items[mid..].iter().map(|x| (x.1.clone(), x.2, x.3)).collect();

        let mut oldb = Block::load(&self.block_device, leaf_pblk as usize * bs, bs);
        self.dx_write_leaf(&mut oldb, &lower, dir_ino, gen);
        oldb.sync_blk_to_disk(&self.block_device);

        let mut newb = Block::load(&self.block_device, new_pblk as usize * bs, bs);
        self.dx_write_leaf(&mut newb, &upper, dir_ino, gen);
        newb.sync_blk_to_disk(&self.block_device);

        Ok((split_hash, new_lblk))
    }

    /// Add an entry to an HTree-indexed directory: descend to the target leaf
    /// and insert; if the leaf is full, split it, register the new leaf in the
    /// index (splitting nodes / growing depth as needed), then insert.
    pub fn dx_add_entry(
        &self,
        parent: &mut Ext4InodeRef,
        child_inode: u32,
        name: &str,
        de_type: DirEntryType,
    ) -> Result<usize> {
        let bs = self.block_size();
        let gen = parent.inode.generation();

        let hash = self.dx_name_hash(parent, name)?;
        let (frames, leaf_lblk) = self.dx_probe_frames_hash(parent, hash)?;
        let leaf_pblk = self.get_pblock_idx(parent, leaf_lblk)?;
        let mut leaf = Block::load(&self.block_device, leaf_pblk as usize * bs, bs);
        if self
            .try_insert_to_existing_block(&mut leaf, name, child_inode, de_type)
            .is_ok()
        {
            self.dir_set_csum(&mut leaf, parent.inode_num, gen);
            leaf.sync_blk_to_disk(&self.block_device);
            return Ok(EOK);
        }

        // Leaf full: split it and register the new leaf in its parent index
        // block, then descend afresh and insert into the correct half.
        let (split_hash, new_leaf) = self.dx_split_leaf(parent, leaf_lblk)?;
        let last = frames.len() - 1;
        self.dx_add_to_index(parent, &frames, last, split_hash, new_leaf)?;

        let leaf_lblk2 = self.dx_probe(parent, name)?;
        let leaf_pblk2 = self.get_pblock_idx(parent, leaf_lblk2)?;
        let mut leaf2 = Block::load(&self.block_device, leaf_pblk2 as usize * bs, bs);
        self.try_insert_to_existing_block(&mut leaf2, name, child_inode, de_type)?;
        self.dir_set_csum(&mut leaf2, parent.inode_num, gen);
        leaf2.sync_blk_to_disk(&self.block_device);
        Ok(EOK)
    }
}
