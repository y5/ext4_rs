//! POSIX access ACLs.
//!
//! An access ACL is stored as the `system.posix_acl_access` extended attribute,
//! whose value is a little-endian `posix_acl_xattr_header` (a u32 version)
//! followed by 8-byte entries `{ e_tag: u16, e_perm: u16, e_id: u32 }`. When a
//! file carries one, the permission check follows the POSIX algorithm instead
//! of the plain owner/group/other mode bits.

use crate::ext4_defs::*;
use crate::prelude::*;

const POSIX_ACL_XATTR_VERSION: u32 = 2;
const ACL_ACCESS_NAME: &str = "system.posix_acl_access";
const ACL_DEFAULT_NAME: &str = "system.posix_acl_default";

// Mode permission masks (the POSIX `S_IRWX{U,G,O}`).
const S_IRWXU: u16 = 0o700;
const S_IRWXG: u16 = 0o070;
const S_IRWXO: u16 = 0o007;
const S_IRWXUGO: u16 = 0o777;

// Entry tags.
const ACL_USER_OBJ: u16 = 0x01;
const ACL_USER: u16 = 0x02;
const ACL_GROUP_OBJ: u16 = 0x04;
const ACL_GROUP: u16 = 0x08;
const ACL_MASK: u16 = 0x10;
const ACL_OTHER: u16 = 0x20;

// Permission bits — identical to R_OK/W_OK/X_OK.
const ACL_RWX: u16 = 0o7;

#[derive(Clone, Copy)]
struct AclEntry {
    tag: u16,
    perm: u16,
    id: u32,
}

/// Parse an access-ACL xattr value into entries, or None if it is malformed or
/// has an unexpected version.
fn parse_acl(value: &[u8]) -> Option<Vec<AclEntry>> {
    if value.len() < 4 {
        return None;
    }
    let version = u32::from_le_bytes([value[0], value[1], value[2], value[3]]);
    if version != POSIX_ACL_XATTR_VERSION {
        return None;
    }
    let body = &value[4..];
    if body.len() % 8 != 0 {
        return None;
    }
    let mut entries = Vec::with_capacity(body.len() / 8);
    for chunk in body.chunks_exact(8) {
        entries.push(AclEntry {
            tag: u16::from_le_bytes([chunk[0], chunk[1]]),
            perm: u16::from_le_bytes([chunk[2], chunk[3]]),
            id: u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
        });
    }
    Some(entries)
}

/// Serialize ACL entries back into a `posix_acl_xattr` value (the inverse of
/// `parse_acl`): a little-endian version header followed by 8-byte entries.
fn serialize_acl(entries: &[AclEntry]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + entries.len() * 8);
    v.extend_from_slice(&POSIX_ACL_XATTR_VERSION.to_le_bytes());
    for e in entries {
        v.extend_from_slice(&e.tag.to_le_bytes());
        v.extend_from_slice(&e.perm.to_le_bytes());
        v.extend_from_slice(&e.id.to_le_bytes());
    }
    v
}

/// Clamp a cloned default ACL against the creation `mode` and rewrite the
/// mode's permission bits, following the kernel `posix_acl_create_masq`.
///
/// `USER_OBJ`/`OTHER` are clamped to the mode's owner/other bits; the `MASK`
/// (or, lacking one, `GROUP_OBJ`) is clamped to the group bits; the mode's
/// u/g/o bits are reduced to the clamped entries. Returns whether the ACL is
/// "not equivalent" to plain mode bits — i.e. has a named user/group or a mask
/// entry, in which case it must be stored as an access ACL.
fn posix_acl_create_masq(entries: &mut [AclEntry], mode: &mut u16) -> bool {
    let mut m = *mode & S_IRWXUGO;
    let mut not_equiv = false;
    let mut group_obj: Option<usize> = None;
    let mut mask_obj: Option<usize> = None;

    for i in 0..entries.len() {
        match entries[i].tag {
            ACL_USER_OBJ => {
                entries[i].perm &= (m >> 6) | !S_IRWXO;
                m &= (entries[i].perm << 6) | !S_IRWXU;
            }
            ACL_USER | ACL_GROUP => not_equiv = true,
            ACL_GROUP_OBJ => group_obj = Some(i),
            ACL_OTHER => {
                entries[i].perm &= m | !S_IRWXO;
                m &= entries[i].perm | !S_IRWXO;
            }
            ACL_MASK => {
                mask_obj = Some(i);
                not_equiv = true;
            }
            _ => {}
        }
    }

    if let Some(mi) = mask_obj {
        entries[mi].perm &= (m >> 3) | !S_IRWXO;
        m &= (entries[mi].perm << 3) | !S_IRWXG;
    } else if let Some(gi) = group_obj {
        entries[gi].perm &= (m >> 3) | !S_IRWXO;
        m &= (entries[gi].perm << 3) | !S_IRWXG;
    }
    // (No GROUP_OBJ and no MASK is a malformed ACL; leave the group bits as-is.)

    *mode = (*mode & !S_IRWXUGO) | (m & S_IRWXUGO);
    not_equiv
}

/// POSIX ACL permission check (`posix_acl_permission`). `want` is the requested
/// R/W/X bits. Returns whether access is granted. The single requesting group
/// stands in for the kernel's `in_group_p` group-set test.
fn acl_permission(
    entries: &[AclEntry],
    file_uid: u16,
    file_gid: u16,
    req_uid: u16,
    req_gid: u16,
    want: u16,
) -> bool {
    let want = want & ACL_RWX;
    let _ = file_gid;

    // The ACL_MASK entry (if any) limits the named-user/group and group-obj
    // entries.
    let mask = entries.iter().find(|e| e.tag == ACL_MASK).map(|e| e.perm);

    let mut found_group = false;
    for e in entries {
        match e.tag {
            ACL_USER_OBJ => {
                if req_uid == file_uid {
                    return e.perm & want == want;
                }
            }
            ACL_USER => {
                if e.id == req_uid as u32 {
                    let effective = mask.map_or(e.perm, |m| e.perm & m);
                    return effective & want == want;
                }
            }
            ACL_GROUP_OBJ => {
                if req_gid == file_gid {
                    found_group = true;
                    let effective = mask.map_or(e.perm, |m| e.perm & m);
                    if effective & want == want {
                        return true;
                    }
                }
            }
            ACL_GROUP => {
                if e.id == req_gid as u32 {
                    found_group = true;
                    let effective = mask.map_or(e.perm, |m| e.perm & m);
                    if effective & want == want {
                        return true;
                    }
                }
            }
            ACL_OTHER => {
                // A group entry matched but didn't grant the access: deny rather
                // than fall through to "other".
                if found_group {
                    return false;
                }
                return e.perm & want == want;
            }
            _ => {}
        }
    }
    false
}

impl Ext4 {
    /// If the inode carries an access ACL, decide whether `want` (R/W/X bits) is
    /// granted for the given requester; None means there is no ACL and the
    /// caller should fall back to the mode bits.
    pub fn acl_access_check(&self, ino: u32, req_uid: u16, req_gid: u16, want: u16) -> Option<bool> {
        let value = self.xattr_get(ino, ACL_ACCESS_NAME).ok()?;
        let entries = parse_acl(&value)?;
        let inode = self.get_inode_ref(ino).inode;
        Some(acl_permission(
            &entries,
            inode.uid,
            inode.gid,
            req_uid,
            req_gid,
            want,
        ))
    }

    /// Apply POSIX creation semantics to a freshly created child (`posix_acl_create`).
    ///
    /// If the parent directory has a `system.posix_acl_default`, it is inherited:
    /// the child's mode is clamped to the default ACL, the masq'd ACL is stored
    /// as the child's access ACL when non-trivial, and a child *directory* also
    /// inherits the default ACL verbatim so it propagates down the tree. When the
    /// parent has no default ACL, the child's mode is masked by `umask` instead.
    /// Symlinks inherit nothing and ignore umask.
    pub fn posix_acl_create(&self, parent_ino: u32, child_ino: u32, umask: u16) -> Result<()> {
        let mut child = self.get_inode_ref(child_ino);
        if child.inode.is_link() {
            return Ok(());
        }
        let is_dir = child.inode.is_dir();
        // Work on the raw permission bits: `InodePerm` only names the owner rwx
        // and setuid/setgid bits, so round-tripping through it would drop the
        // group/other bits. Preserve the type and special (setuid/setgid/sticky)
        // bits; only the low 9 rwx bits change.
        let orig_mode = child.inode.mode();
        let mut perm = orig_mode & S_IRWXUGO;

        let default_val = self.xattr_get(parent_ino, ACL_DEFAULT_NAME).ok();
        let default_entries = default_val.as_deref().and_then(parse_acl);

        match default_entries {
            None => {
                // No (valid) default ACL: the umask governs the new mode.
                perm &= !(umask & S_IRWXUGO);
                child.inode.set_mode((orig_mode & !S_IRWXUGO) | perm);
                self.write_back_inode(&mut child);
            }
            Some(mut entries) => {
                // A default ACL is present; umask is not applied.
                let not_equiv = posix_acl_create_masq(&mut entries, &mut perm);
                child.inode.set_mode((orig_mode & !S_IRWXUGO) | (perm & S_IRWXUGO));
                self.write_back_inode(&mut child);

                if not_equiv {
                    self.xattr_set(child_ino, ACL_ACCESS_NAME, &serialize_acl(&entries), 0)?;
                }
                if is_dir {
                    // Propagate the parent's default ACL unchanged.
                    self.xattr_set(child_ino, ACL_DEFAULT_NAME, default_val.as_ref().unwrap(), 0)?;
                }
            }
        }
        Ok(())
    }
}
