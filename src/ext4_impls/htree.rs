//! ext4 HTree (hashed B-tree) directory indexing.
//!
//! This module ports the directory hash from `fs/ext4/hash.c` and (in later
//! phases) the dx_root/dx_node index traversal and maintenance.

use crate::prelude::*;

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
