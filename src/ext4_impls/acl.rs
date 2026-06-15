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
}
