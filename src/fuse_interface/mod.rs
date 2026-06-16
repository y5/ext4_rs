use crate::prelude::*;

use crate::ext4_defs::*;
use crate::return_errno;
use crate::return_errno_with_message;
use crate::utils::path_check;

// export some definitions
pub use crate::ext4_defs::BlockDevice;
pub use crate::ext4_defs::Ext4;
pub use crate::ext4_defs::InodeFileType;
pub use crate::ext4_defs::BLOCK_SIZE;
pub use crate::ext4_defs::{FileLock, F_RDLCK, F_UNLCK, F_WRLCK};
pub use crate::ext4_defs::{Ext4DirEntry, Ext4DirSearchResult};

/// fuser interface for ext4
impl Ext4 {
    /// Look up a directory entry by name and get its attributes.
    pub fn fuse_lookup(&self, parent: u64, name: &str) -> Result<FileAttr> {
        let mut search_result = Ext4DirSearchResult::new(Ext4DirEntry::default());

        self.dir_find_entry(parent as u32, name, &mut search_result)?;

        let inode_num = search_result.dentry.inode;

        let inode_ref = self.get_inode_ref(inode_num);
        let file_attr = FileAttr::from_inode_ref(&inode_ref, self.block_size() as u32);

        Ok(file_attr)
    }

    /// Get file attributes.
    pub fn fuse_getattr(&self, ino: u64) -> Result<FileAttr> {
        let inode_ref = self.get_inode_ref(ino as u32);
        let file_attr = FileAttr::from_inode_ref(&inode_ref, self.block_size() as u32);
        Ok(file_attr)
    }

    /// Set file attributes.
    pub fn fuse_setattr(
        &self,
        ino: u64,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        atime: Option<u32>,
        mtime: Option<u32>,
        ctime: Option<u32>,
        fh: Option<u64>,
        crtime: Option<u32>,
        chgtime: Option<u32>,
        bkuptime: Option<u32>,
        flags: Option<u32>,
    ) {
        let mut inode_ref = self.get_inode_ref(ino as u32);

        let mut attr = FileAttr::default();

        if let Some(mode) = mode {
            let inode_file_type =
                InodeFileType::from_bits(mode as u16 & EXT4_INODE_MODE_TYPE_MASK).unwrap();
            attr.kind = inode_file_type;
            let inode_perm = InodePerm::from_bits(mode as u16 & EXT4_INODE_MODE_PERM_MASK).unwrap();
            attr.perm = inode_perm;
        }

        if let Some(uid) = uid {
            attr.uid = uid
        }

        if let Some(gid) = gid {
            attr.gid = gid
        }

        if let Some(size) = size {
            attr.size = size
        }

        if let Some(atime) = atime {
            attr.atime = atime
        }

        if let Some(mtime) = mtime {
            attr.mtime = mtime
        }

        if let Some(ctime) = ctime {
            attr.ctime = ctime
        }

        if let Some(crtime) = crtime {
            attr.crtime = crtime
        }

        if let Some(chgtime) = chgtime {
            attr.chgtime = chgtime
        }

        if let Some(bkuptime) = bkuptime {
            attr.bkuptime = bkuptime
        }

        if let Some(flags) = flags {
            attr.flags = flags
        }

        inode_ref.set_attr(&attr);

        self.write_back_inode(&mut inode_ref);
    }

    /// Read symbolic link.
    pub fn fuse_readlink(&mut self, ino: u64) -> Result<Vec<u8>> {
        let inode_ref = self.get_inode_ref(ino as u32);
        let file_size = inode_ref.inode.size() as usize;

        // Fast symlink: the target lives inline in the inode block area with no
        // data blocks allocated. read_at would misread the block array as an
        // extent tree, so unpack it directly here.
        if inode_ref.inode.blocks_count() == 0 {
            let block = inode_ref.inode.block();
            let mut raw = [0u8; 15 * 4];
            for (i, word) in block.iter().enumerate() {
                raw[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            return Ok(raw[..file_size].to_vec());
        }

        // Slow symlink: the target is stored as ordinary file data.
        let mut read_buf = vec![0; file_size];
        let read_size = self.read_at(ino as u32, 0, &mut read_buf)?;
        Ok(read_buf)
    }

    /// Create a regular file, character device, block device, fifo or socket node.
    pub fn fuse_mknod(
        &self,
        parent: u64,
        name: &str,
        mode: u32,
        umask: u32,
        rdev: u32,
    ) -> Result<Ext4InodeRef> {
        self.journaled(|| self.fuse_mknod_impl(parent, name, mode, umask, rdev))
    }

    fn fuse_mknod_impl(
        &self,
        parent: u64,
        name: &str,
        mode: u32,
        umask: u32,
        rdev: u32,
    ) -> Result<Ext4InodeRef> {
        let mut search_result = Ext4DirSearchResult::new(Ext4DirEntry::default());
        let r = self.dir_find_entry(parent as u32, name, &mut search_result);
        if r.is_ok() {
            return_errno!(Errno::EEXIST);
        }
        let inode_ref =
            self.create_special_with_attr(parent as u32, name, mode as u16, rdev, 0, 0)?;
        self.posix_acl_create(parent as u32, inode_ref.inode_num, umask as u16)?;
        Ok(inode_ref)
    }

    /// Create a regular file, character device, block device, fifo or socket node.
    pub fn fuse_mknod_with_attr(
        &self,
        parent: u64,
        name: &str,
        mode: u32,
        umask: u32,
        rdev: u32,
        uid: u32,
        gid: u32,
    ) -> Result<Ext4InodeRef> {
        self.journaled(|| self.fuse_mknod_with_attr_impl(parent, name, mode, umask, rdev, uid, gid))
    }

    fn fuse_mknod_with_attr_impl(
        &self,
        parent: u64,
        name: &str,
        mode: u32,
        umask: u32,
        rdev: u32,
        uid: u32,
        gid: u32,
    ) -> Result<Ext4InodeRef> {
        let mut search_result = Ext4DirSearchResult::new(Ext4DirEntry::default());
        let r = self.dir_find_entry(parent as u32, name, &mut search_result);
        if r.is_ok() {
            return_errno!(Errno::EEXIST);
        }
        let inode_ref = self.create_special_with_attr(
            parent as u32,
            name,
            mode as u16,
            rdev,
            uid as u16,
            gid as u16,
        )?;
        self.posix_acl_create(parent as u32, inode_ref.inode_num, umask as u16)?;
        Ok(inode_ref)
    }

    /// Create a directory.
    pub fn fuse_mkdir(&mut self, parent: u64, name: &str, mode: u32, umask: u32) -> Result<usize> {
        self.journaled_mut(|s| s.fuse_mkdir_impl(parent, name, mode, umask))
    }

    fn fuse_mkdir_impl(&mut self, parent: u64, name: &str, mode: u32, umask: u32) -> Result<usize> {
        let mut search_result = Ext4DirSearchResult::new(Ext4DirEntry::default());
        let r = self.dir_find_entry(parent as u32, name, &mut search_result);
        if r.is_ok() {
            return_errno!(Errno::EEXIST);
        }
        let file_type = InodeFileType::from_bits(mode as u16).unwrap();
        if file_type != InodeFileType::S_IFDIR {
            // The mode is not a directory
            return_errno_with_message!(Errno::EINVAL, "Invalid mode for directory creation");
        }
        let inode_ref = self.create(parent as u32, name, mode as u16)?;
        self.posix_acl_create(parent as u32, inode_ref.inode_num, umask as u16)?;
        Ok(EOK)
    }

    /// Create a directory.
    pub fn fuse_mkdir_with_attr(
        &mut self,
        parent: u64,
        name: &str,
        mode: u32,
        umask: u32,
        uid: u32,
        gid: u32,
    ) -> Result<Ext4InodeRef> {
        self.journaled_mut(|s| s.fuse_mkdir_with_attr_impl(parent, name, mode, umask, uid, gid))
    }

    fn fuse_mkdir_with_attr_impl(
        &mut self,
        parent: u64,
        name: &str,
        mode: u32,
        umask: u32,
        uid: u32,
        gid: u32,
    ) -> Result<Ext4InodeRef> {
        let mut search_result = Ext4DirSearchResult::new(Ext4DirEntry::default());
        let r = self.dir_find_entry(parent as u32, name, &mut search_result);
        if r.is_ok() {
            return_errno!(Errno::EEXIST);
        }

        // mkdir via fuse passes a mode of 0. so we need to set default mode
        let file_type = match InodeFileType::from_bits(mode as u16) {
            Some(file_type) => file_type,
            None => InodeFileType::S_IFDIR,
        };
        let mode = file_type.bits();
        let inode_ref = self.create_with_attr(parent as u32, name, mode, uid as u16, gid as u16)?;
        self.posix_acl_create(parent as u32, inode_ref.inode_num, umask as u16)?;

        Ok(inode_ref)
    }

    /// Remove a file.
    pub fn fuse_unlink(&self, parent: u64, name: &str) -> Result<usize> {
        self.journaled(|| self.fuse_unlink_impl(parent, name))
    }

    fn fuse_unlink_impl(&self, parent: u64, name: &str) -> Result<usize> {
        // unlink actual remove a file

        // get child inode num
        let mut parent_inode = parent as u32;
        let mut nameoff = 0;
        let child_inode = self.generic_open(name, &mut parent_inode, false, 0, &mut nameoff)?;

        let mut child_inode_ref = self.get_inode_ref(child_inode);
        let child_link_cnt = child_inode_ref.inode.links_count();
        if child_link_cnt == 1 {
            self.truncate_inode(&mut child_inode_ref, 0)?;
        }

        // get child name
        let mut is_goal = false;
        let p = &name[nameoff as usize..];
        let len = path_check(p, &mut is_goal);

        // load parent
        let mut parent_inode_ref = self.get_inode_ref(parent_inode);

        let r = self.unlink(&mut parent_inode_ref, &mut child_inode_ref, &p[..len])?;

        Ok(EOK)
    }
    /// Remove a directory.
    pub fn fuse_rmdir(&mut self, parent: u64, name: &str) -> Result<usize> {
        self.journaled_mut(|s| s.fuse_rmdir_impl(parent, name))
    }

    fn fuse_rmdir_impl(&mut self, parent: u64, name: &str) -> Result<usize> {
        // Delegate to the dir_remove primitive, which removes the entry, frees
        // the empty directory's inode, and drops the parent's '..' back-link.
        // The previous hand-rolled body left the child inode allocated and the
        // parent link count too high (the `to do` notes below were never done),
        // so e2fsck saw an unconnected inode and a wrong reference count.
        self.dir_remove(parent as u32, name)
    }
    /// Create a symbolic link.
    pub fn fuse_symlink(&mut self, parent: u64, link_name: &str, target: &str) -> Result<usize> {
        self.journaled_mut(|s| s.fuse_symlink_impl(parent, link_name, target))
    }

    fn fuse_symlink_impl(&mut self, parent: u64, link_name: &str, target: &str) -> Result<usize> {
        let mut search_result = Ext4DirSearchResult::new(Ext4DirEntry::default());
        let r = self.dir_find_entry(parent as u32, link_name, &mut search_result);
        if r.is_ok() {
            return_errno!(Errno::EEXIST);
        }

        let mut mode = 0o777;
        let file_type = InodeFileType::S_IFLNK;
        mode |= file_type.bits();

        let inode_ref = self.create(parent as u32, link_name, mode)?;
        self.write_symlink_target(inode_ref.inode_num, target)?;

        Ok(EOK)
    }

    /// Store a symlink's target on disk. Targets shorter than 60 bytes are kept
    /// inline in the inode's block area ("fast symlink", no data blocks);
    /// longer ones spill into a data block ("slow symlink"). This matches what
    /// the kernel writes and what e2fsck expects.
    fn write_symlink_target(&self, ino: u32, target: &str) -> Result<()> {
        let bytes = target.as_bytes();

        // The inode block area is 15 * 4 = 60 bytes.
        const INLINE_CAP: usize = 15 * 4;

        if bytes.len() < INLINE_CAP {
            let mut inode_ref = self.get_inode_ref(ino);

            // create() set the extents flag and an extent header in the block
            // area; a fast symlink uses neither, and e2fsck rejects EXTENT_FL
            // on a fast symlink.
            let flags = inode_ref.inode.flags() & !(EXT4_INODE_FLAG_EXTENTS as u32);
            inode_ref.inode.set_flags(flags);

            // Pack the target bytes (little-endian, on-disk order) into the
            // 15-word block array, zero-padding the rest.
            let mut raw = [0u8; INLINE_CAP];
            raw[..bytes.len()].copy_from_slice(bytes);
            let mut block = [0u32; 15];
            for (i, word) in block.iter_mut().enumerate() {
                let b = i * 4;
                *word = u32::from_le_bytes([raw[b], raw[b + 1], raw[b + 2], raw[b + 3]]);
            }
            inode_ref.inode.set_block(block);
            inode_ref.inode.set_size(bytes.len() as u64);
            inode_ref.inode.set_blocks_count(0);

            self.write_back_inode(&mut inode_ref);
        } else {
            // Slow symlink: keep the extent tree create() set up and write the
            // target as ordinary file data; write_at allocates the block and
            // records the size.
            self.write_at(ino, 0, bytes)?;
        }

        Ok(())
    }
    /// Create a hard link.
    /// Params:
    /// ino: the inode number of the source file
    /// newparent: the inode number of the new parent directory
    /// newname: the name of the new file
    ///
    ///
    pub fn fuse_link(&mut self, ino: u64, newparent: u64, newname: &str) -> Result<usize> {
        self.journaled_mut(|s| s.fuse_link_impl(ino, newparent, newname))
    }

    fn fuse_link_impl(&mut self, ino: u64, newparent: u64, newname: &str) -> Result<usize> {
        let mut parent_inode_ref = self.get_inode_ref(newparent as u32);
        let mut child_inode_ref = self.get_inode_ref(ino as u32);

        // to do if child already exists we should not add . and .. in child directory
        self.link(&mut parent_inode_ref, &mut child_inode_ref, newname)?;

        // link() only bumps links_count in memory; persist both inodes (with
        // checksums) so the on-disk link count matches the directory entries,
        // mirroring what create() does after its own link() call.
        self.write_back_inode(&mut parent_inode_ref);
        self.write_back_inode(&mut child_inode_ref);

        Ok(EOK)
    }

    /// Open a file.
    /// Open flags (with the exception of O_CREAT, O_EXCL, O_NOCTTY and O_TRUNC) are
    /// available in flags. Filesystem may store an arbitrary file handle (pointer, index,
    /// etc) in fh, and use this in other all other file operations (read, write, flush,
    /// release, fsync). Filesystem may also implement stateless file I/O and not store
    /// anything in fh. There are also some flags (direct_io, keep_cache) which the
    /// filesystem may set, to change the way the file is opened. See fuse_file_info
    /// structure in <fuse_common.h> for more details.
    pub fn fuse_open(&mut self, ino: u64, flags: i32) -> Result<usize> {
        let inode_ref = self.get_inode_ref(ino as u32);

        // check permission
        let file_type = inode_ref.inode.file_type();
        let file_perm = inode_ref.inode.file_perm();

        let can_read = file_perm.contains(InodePerm::S_IREAD);
        let can_write = file_perm.contains(InodePerm::S_IWRITE);
        let can_execute = file_perm.contains(InodePerm::S_IEXEC);

        // If trying to open the file in write mode, check for write permissions
        if ((flags & O_WRONLY != 0) || (flags & O_RDWR != 0)) && !can_write {
            return_errno_with_message!(Errno::EACCES, "Permission denied can not write");
        }
        // If trying to open the file in read mode, check for read permissions
        if ((flags & O_RDONLY != 0) || (flags & O_RDWR != 0)) && !can_read {
            return_errno_with_message!(Errno::EACCES, "Permission denied can not read");
        }

        // If trying to open the file in read mode, check for read permissions
        if ((flags & O_EXCL != 0) || (flags & O_RDWR != 0)) && !can_execute {
            return_errno_with_message!(Errno::EACCES, "Permission denied can not exec");
        }

        Ok(EOK)
    }

    /// Read data.
    /// Read should send exactly the number of bytes requested except on EOF or error,
    /// otherwise the rest of the data will be substituted with zeroes. An exception to
    /// this is when the file has been opened in 'direct_io' mode, in which case the
    /// return value of the read system call will reflect the return value of this
    /// operation. fh will contain the value set by the open method, or will be undefined
    /// if the open method didn't set any value.
    ///
    /// flags: these are the file flags, such as O_SYNC. Only supported with ABI >= 7.9
    /// lock_owner: only supported with ABI >= 7.9
    pub fn fuse_read(
        &self,
        ino: u64,
        fh: u64,
        offset: i64,
        size: u32,
        flags: i32,
        lock_owner: Option<u64>,
    ) -> Result<Vec<u8>> {
        let mut data = vec![0u8; size as usize];
        let read_size = self.read_at(ino as u32, offset as usize, &mut data)?;
        let r = data[..read_size].to_vec();
        Ok(r)
    }

    /// Write data.
    /// Write should return exactly the number of bytes requested except on error. An
    /// exception to this is when the file has been opened in 'direct_io' mode, in
    /// which case the return value of the write system call will reflect the return
    /// value of this operation. fh will contain the value set by the open method, or
    /// will be undefined if the open method didn't set any value.
    ///
    /// write_flags: will contain FUSE_WRITE_CACHE, if this write is from the page cache. If set,
    /// the pid, uid, gid, and fh may not match the value that would have been sent if write cachin
    /// is disabled
    /// flags: these are the file flags, such as O_SYNC. Only supported with ABI >= 7.9
    /// lock_owner: only supported with ABI >= 7.9
    pub fn fuse_write(
        &self,
        ino: u64,
        fh: u64,
        offset: i64,
        data: &[u8],
        write_flags: u32,
        flags: i32,
        lock_owner: Option<u64>,
    ) -> Result<usize> {
        let write_size = self.write_at(ino as u32, offset as usize, data)?;
        Ok(write_size)
    }

    /// Open a directory.
    /// Filesystem may store an arbitrary file handle (pointer, index, etc) in fh, and
    /// use this in other all other directory stream operations (readdir, releasedir,
    /// fsyncdir). Filesystem may also implement stateless directory I/O and not store
    /// anything in fh, though that makes it impossible to implement standard conforming
    /// directory stream operations in case the contents of the directory can change
    /// between opendir and releasedir.
    pub fn fuse_opendir(&mut self, ino: u64, flags: i32) -> Result<usize> {
        let inode_ref = self.get_inode_ref(ino as u32);

        // 检查是否为目录
        if !inode_ref.inode.is_dir() {
            return_errno_with_message!(Errno::ENOTDIR, "Not a directory");
        }

        // // 检查权限（例如，只允许读取目录）
        // let file_perm = inode_ref.inode.file_perm();
        // if !file_perm.contains(InodePerm::S_IREAD) {
        //     return_errno_with_message!(Errno::EACCES, "Permission denied");
        // }

        // 成功打开目录，返回文件句柄（这里假设返回 inode 编号作为文件句柄）
        Ok(ino as usize)
    }

    /// Read directory.
    /// Send a buffer filled using buffer.fill(), with size not exceeding the
    /// requested size. Send an empty buffer on end of stream. fh will contain the
    /// value set by the opendir method, or will be undefined if the opendir method
    /// didn't set any value.
    pub fn fuse_readdir(
        &self,
        ino: u64,
        fh: u64,
        offset: i64,
    ) -> Result<Vec<Ext4DirEntryWithOffset>> {
        // `offset` is the opaque resume cookie a previous readdir handed back
        // (the byte position of the next entry), or 0 to start from the top.
        Ok(self.dir_entries_with_offset_from(ino as u32, offset as u64))
    }

    /// Create and open a file.
    /// If the file does not exist, first create it with the specified mode, and then
    /// open it. Open flags (with the exception of O_NOCTTY) are available in flags.
    /// Filesystem may store an arbitrary file handle (pointer, index, etc) in fh,
    /// and use this in other all other file operations (read, write, flush, release,
    /// fsync). There are also some flags (direct_io, keep_cache) which the
    /// filesystem may set, to change the way the file is opened. See fuse_file_info
    /// structure in <fuse_common.h> for more details. If this method is not
    /// implemented or under Linux kernel versions earlier than 2.6.15, the mknod()
    /// and open() methods will be called instead.
    pub fn fuse_create(
        &mut self,
        parent: u64,
        name: &str,
        mode: u32,
        umask: u32,
        flags: i32,
    ) -> Result<usize> {
        self.journaled_mut(|s| s.fuse_create_impl(parent, name, mode, umask, flags))
    }

    fn fuse_create_impl(
        &mut self,
        parent: u64,
        name: &str,
        mode: u32,
        umask: u32,
        flags: i32,
    ) -> Result<usize> {
        // check file exist
        let mut search_result = Ext4DirSearchResult::new(Ext4DirEntry::default());
        let r = self.dir_find_entry(parent as u32, name, &mut search_result);
        if r.is_ok() {
            let inode_ref = self.get_inode_ref(search_result.dentry.inode);

            // check permission
            let file_perm = inode_ref.inode.file_perm();

            let can_read = file_perm.contains(InodePerm::S_IREAD);
            let can_write = file_perm.contains(InodePerm::S_IWRITE);
            let can_execute = file_perm.contains(InodePerm::S_IEXEC);

            // If trying to open the file in write mode, check for write permissions
            if ((flags & O_WRONLY != 0) || (flags & O_RDWR != 0)) && !can_write {
                return_errno_with_message!(Errno::EACCES, "Permission denied can not write");
            }

            // If trying to open the file in read mode, check for read permissions
            if (flags & O_RDONLY != 0) || (flags & O_RDWR != 0) && !can_read {
                return_errno_with_message!(Errno::EACCES, "Permission denied can not read");
            }

            // If trying to open the file in read mode, check for read permissions
            if (flags & O_EXCL != 0) || (flags & O_RDWR != 0) && !can_execute {
                return_errno_with_message!(Errno::EACCES, "Permission denied can not exec");
            }

            return Ok(EOK);
        } else {
            //create file
            let inode_ref = self.create(parent as u32, name, mode as u16)?;
            self.posix_acl_create(parent as u32, inode_ref.inode_num, umask as u16)?;
        }

        Ok(EOK)
    }

    /// Check file access permissions.
    /// This will be called for the access() system call. If the 'default_permissions'
    /// mount option is given, this method is not called. This method is not called
    /// under Linux kernel versions 2.4.x
    /// int access(const char *pathname, int mode);
    /// int faccessat(int dirfd, const char *pathname, int mode, int flags);
    ///
    /// uid and gid come from request
    pub fn fuse_access(&mut self, ino: u64, uid: u16, gid: u16, mode: u16, mask: i32) -> bool {
        // An access ACL, when present, overrides the plain mode-bit check.
        let want = mode & 0o7;
        if let Some(granted) = self.acl_access_check(ino as u32, uid, gid, want) {
            return granted;
        }

        let inode_ref = self.get_inode_ref(ino as u32);
        inode_ref.inode.check_access(uid, gid, mode, mask as u16)
    }

    /// Get file system statistics.
    /// Linux stat syscall defines:
    /// int stat(const char *restrict pathname, struct stat *restrict statbuf);
    /// int fstatat(int dirfd, const char *restrict pathname, struct stat *restrict statbuf, int flags);
    pub fn fuse_statfs(&mut self, ino: u64) -> Result<LinuxStat> {
        let inode_ref = self.get_inode_ref(ino as u32);
        let linux_stat = LinuxStat::from_inode_ref(&inode_ref, self.block_size() as u32);
        Ok(linux_stat)
    }

    /// Initialize filesystem.
    /// Called before any other filesystem method.
    /// The kernel module connection can be configured using the KernelConfig object
    pub fn fuse_init(&mut self) -> Result<usize> {
        Ok(EOK)
    }

    /// Clean up filesystem.
    /// Called on filesystem exit.
    pub fn fuse_destroy(&mut self) -> Result<usize> {
        Ok(EOK)
    }

    /// Rename a file.
    pub fn fuse_rename(
        &mut self,
        parent: u64,
        name: &str,
        newparent: u64,
        newname: &str,
        flags: u32,
    ) -> Result<usize> {
        self.journaled_mut(|s| s.fuse_rename_impl(parent, name, newparent, newname, flags))
    }

    fn fuse_rename_impl(
        &mut self,
        parent: u64,
        name: &str,
        newparent: u64,
        newname: &str,
        flags: u32,
    ) -> Result<usize> {
        let old_parent = parent as u32;
        let new_parent = newparent as u32;

        // Source must exist.
        let mut src = Ext4DirSearchResult::new(Ext4DirEntry::default());
        self.dir_find_entry(old_parent, name, &mut src)?;
        let src_ino = src.dentry.inode;
        let src_ref = self.get_inode_ref(src_ino);
        let src_is_dir = src_ref.inode.is_dir();

        const RENAME_NOREPLACE: u32 = 1;
        const RENAME_EXCHANGE: u32 = 2;

        // A directory cannot be moved into itself or one of its descendants -
        // that would detach the subtree into a loop.
        if src_is_dir && self.dir_is_ancestor(src_ino, new_parent) {
            return_errno!(Errno::EINVAL);
        }

        // RENAME_EXCHANGE atomically swaps two existing names; handled wholly
        // on its own since it frees nothing and touches both entries.
        if flags & RENAME_EXCHANGE != 0 {
            return self.rename_exchange(old_parent, name, src_ino, src_is_dir, new_parent, newname);
        }

        // Examine the destination name, if it already exists.
        let mut dst = Ext4DirSearchResult::new(Ext4DirEntry::default());
        if self.dir_find_entry(new_parent, newname, &mut dst).is_ok() {
            // RENAME_NOREPLACE refuses to clobber an existing destination.
            if flags & RENAME_NOREPLACE != 0 {
                return_errno!(Errno::EEXIST);
            }

            let dst_ino = dst.dentry.inode;

            // Renaming a name onto itself (same inode) is a no-op.
            if dst_ino == src_ino {
                return Ok(EOK);
            }

            let dst_is_dir = self.get_inode_ref(dst_ino).inode.is_dir();

            // POSIX type compatibility between the two names.
            if dst_is_dir && !src_is_dir {
                return_errno!(Errno::EISDIR);
            }
            if !dst_is_dir && src_is_dir {
                return_errno!(Errno::ENOTDIR);
            }

            // Free the destination name so the source can take it. These checks
            // run before any mutation, so a rejected rename leaves the image
            // untouched.
            if dst_is_dir {
                // dir_remove enforces ENOTEMPTY and drops the removed
                // directory's '..' credit from new_parent.
                self.dir_remove(new_parent, newname)?;
            } else {
                self.fuse_unlink(new_parent as u64, newname)?;
            }
        }

        // Point a new entry at the source inode under the new name. dir_add_entry
        // does not touch link counts, and may grow the parent directory, so the
        // parent inode must be written back afterwards.
        let mut np = self.get_inode_ref(new_parent);
        self.dir_add_entry(&mut np, &src_ref, newname)?;
        self.write_back_inode(&mut np);

        // Drop the old name. dir_remove_entry leaves the inode's link count
        // alone, so the moved inode keeps exactly one reference.
        let mut op = self.get_inode_ref(old_parent);
        self.dir_remove_entry(&mut op, name)?;
        self.write_back_inode(&mut op);

        // Moving a directory to a different parent: its ".." back-reference now
        // points at the wrong directory. Repoint it, and move the subdirectory
        // link credit from the old parent to the new one.
        if src_is_dir && old_parent != new_parent {
            self.dir_set_entry_target(src_ino, "..", new_parent)?;
            self.adjust_dir_links(old_parent, -1)?;
            self.adjust_dir_links(new_parent, 1)?;
        }

        Ok(EOK)
    }

    /// True if `ancestor` is `start` itself or any directory above it, walking
    /// up through '..'. Used to reject moving a directory into its own subtree.
    fn dir_is_ancestor(&self, ancestor: u32, start: u32) -> bool {
        let mut cur = start;
        loop {
            if cur == ancestor {
                return true;
            }
            if cur == ROOT_INODE {
                return false;
            }
            let mut res = Ext4DirSearchResult::new(Ext4DirEntry::default());
            if self.dir_find_entry(cur, "..", &mut res).is_err() {
                return false;
            }
            let parent = res.dentry.inode;
            // Root's '..' points to itself; stop rather than loop forever.
            if parent == cur {
                return false;
            }
            cur = parent;
        }
    }

    /// Add `delta` (which may be negative) to a directory's link count and
    /// persist it. Used to move subdirectory '..' credit between parents.
    fn adjust_dir_links(&self, ino: u32, delta: i64) -> Result<()> {
        if delta == 0 {
            return Ok(());
        }
        let mut r = self.get_inode_ref(ino);
        let new = (r.inode.links_count() as i64 + delta).max(0) as u16;
        r.inode.set_links_count(new);
        self.write_back_inode(&mut r);
        Ok(())
    }

    /// RENAME_EXCHANGE: atomically swap which inode the two names refer to. Both
    /// names must exist; nothing is created or freed. The directory entries stay
    /// in their parents - only their target inode (and stored type) is swapped -
    /// so for directories that change parent we repoint '..' and move the
    /// subdirectory link credit between the parents.
    fn rename_exchange(
        &mut self,
        old_parent: u32,
        name: &str,
        src_ino: u32,
        src_is_dir: bool,
        new_parent: u32,
        newname: &str,
    ) -> Result<usize> {
        // The partner name must exist.
        let mut dst = Ext4DirSearchResult::new(Ext4DirEntry::default());
        self.dir_find_entry(new_parent, newname, &mut dst)?;
        let dst_ino = dst.dentry.inode;

        // Exchanging a name with itself is a no-op.
        if dst_ino == src_ino {
            return Ok(EOK);
        }
        let dst_is_dir = self.get_inode_ref(dst_ino).inode.is_dir();

        // Neither directory may be swapped into its own subtree.
        if dst_is_dir && self.dir_is_ancestor(dst_ino, old_parent) {
            return_errno!(Errno::EINVAL);
        }

        // Swap the targets in place.
        self.dir_set_entry_target(old_parent, name, dst_ino)?;
        self.dir_set_entry_target(new_parent, newname, src_ino)?;

        // When the two names live in different directories, each directory that
        // moved needs its '..' repointed, and each parent's link count adjusted
        // by what it gained minus what it lost.
        if old_parent != new_parent {
            if src_is_dir {
                self.dir_set_entry_target(src_ino, "..", new_parent)?;
            }
            if dst_is_dir {
                self.dir_set_entry_target(dst_ino, "..", old_parent)?;
            }
            self.adjust_dir_links(old_parent, dst_is_dir as i64 - src_is_dir as i64)?;
            self.adjust_dir_links(new_parent, src_is_dir as i64 - dst_is_dir as i64)?;
        }

        Ok(EOK)
    }

    /// Flush method.
    /// This is called on each close() of the opened file. Since file descriptors can
    /// be duplicated (dup, dup2, fork), for one open call there may be many flush
    /// calls. Filesystems shouldn't assume that flush will always be called after some
    /// writes, or that if will be called at all. fh will contain the value set by the
    /// open method, or will be undefined if the open method didn't set any value.
    /// NOTE: the name of the method is misleading, since (unlike fsync) the filesystem
    /// is not forced to flush pending writes. One reason to flush data, is if the
    /// filesystem wants to return write errors. If the filesystem supports file locking
    /// operations (setlk, getlk) it should remove all locks belonging to 'lock_owner'.
    pub fn fuse_flush(&mut self, _ino: u64, _fh: u64, lock_owner: u64) -> Result<usize> {
        // Nothing to flush: writes are write-through, so there is no per-fh dirty
        // buffer to push. We only honor the documented contract of dropping any
        // POSIX locks held by this owner (the lock table is maintained by
        // setlk/getlk).
        self.locks_release_owner(lock_owner);
        Ok(EOK)
    }

    /// Release an open file.
    /// Release is called when there are no more references to an open file: all file
    /// descriptors are closed and all memory mappings are unmapped. For every open
    /// call there will be exactly one release call. The filesystem may reply with an
    /// error, but error values are not returned to close() or munmap() which triggered
    /// the release. fh will contain the value set by the open method, or will be undefined
    /// if the open method didn't set any value. flags will contain the same flags as for
    /// open.
    pub fn fuse_release(
        &mut self,
        _ino: u64,
        _fh: u64,
        _flags: i32,
        lock_owner: Option<u64>,
        _flush: bool,
    ) -> Result<usize> {
        // No per-open state to tear down (file I/O is stateless here). Drop any
        // locks held by the closing owner, as the kernel expects.
        if let Some(owner) = lock_owner {
            self.locks_release_owner(owner);
        }
        Ok(EOK)
    }

    /// Synchronize file contents.
    /// If the datasync parameter is non-zero, then only the user data should be flushed,
    /// not the meta data.
    pub fn fuse_fsync(&mut self, _ino: u64, _fh: u64, _datasync: bool) -> Result<usize> {
        // Data and metadata are already written through to the device; force the
        // device's own buffers to stable storage. We don't separately buffer
        // metadata, so `datasync` makes no difference here.
        self.block_device.flush();
        Ok(EOK)
    }

    /// Read directory.
    /// Send a buffer filled using buffer.fill(), with size not exceeding the
    /// requested size. Send an empty buffer on end of stream. fh will contain the
    /// value set by the opendir method, or will be undefined if the opendir method
    /// didn't set any value.
    pub fn fuse_readdirplus(
        &self,
        ino: u64,
        fh: u64,
        offset: i64,
    ) -> Result<Vec<Ext4DirEntryPlus>> {
        // Same entries and resume cookies as readdir, plus each entry's stat.
        let block_size = self.block_size() as u32;
        let plus = self
            .dir_entries_with_offset_from(ino as u32, offset as u64)
            .into_iter()
            .map(|e| {
                let inode_ref = self.get_inode_ref(e.entry.inode);
                Ext4DirEntryPlus {
                    entry: e.entry,
                    next_offset: e.next_offset,
                    attr: FileAttr::from_inode_ref(&inode_ref, block_size),
                }
            })
            .collect();
        Ok(plus)
    }

    /// Release an open directory.
    /// For every opendir call there will be exactly one releasedir call. fh will
    /// contain the value set by the opendir method, or will be undefined if the
    /// opendir method didn't set any value.
    pub fn fuse_releasedir(&mut self, _ino: u64, _fh: u64, _flags: i32) -> Result<usize> {
        // opendir stores no state to release.
        Ok(EOK)
    }

    /// Drop every advisory lock held by `owner` across all inodes. Called from
    /// `flush`/`release` so a closing file descriptor leaves no stale locks.
    pub(crate) fn locks_release_owner(&mut self, owner: u64) {
        for locks in self.locks.values_mut() {
            locks.retain(|l| l.owner != owner);
        }
        self.locks.retain(|_, locks| !locks.is_empty());
    }

    /// Synchronize directory contents.
    /// If the datasync parameter is set, then only the directory contents should
    /// be flushed, not the meta data. fh will contain the value set by the opendir
    /// method, or will be undefined if the opendir method didn't set any value.
    pub fn fuse_fsyncdir(&mut self, _ino: u64, _fh: u64, _datasync: bool) -> Result<usize> {
        self.block_device.flush();
        Ok(EOK)
    }

    /// Set an extended attribute.
    pub fn fuse_setxattr(
        &mut self,
        ino: u64,
        name: &str,
        value: &[u8],
        flags: i32,
        _position: u32,
    ) -> Result<usize> {
        self.xattr_set(ino as u32, name, value, flags)?;
        Ok(EOK)
    }

    /// Get an extended attribute.
    /// With `size == 0`, returns just the value's length (so the caller can size
    /// its buffer); otherwise returns the value, or ERANGE if it doesn't fit.
    pub fn fuse_getxattr(&mut self, ino: u64, name: &str, size: u32) -> Result<Vec<u8>> {
        let value = self.xattr_get(ino as u32, name)?;
        if size == 0 {
            return Ok(value); // caller inspects the length
        }
        if value.len() > size as usize {
            return_errno!(Errno::ERANGE);
        }
        Ok(value)
    }

    /// List extended attribute names as a NUL-terminated, NUL-separated buffer.
    /// With `size == 0`, returns the buffer so the caller can read its length;
    /// otherwise returns it only if it fits, else ERANGE.
    pub fn fuse_listxattr(&mut self, ino: u64, size: u32) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        for name in self.xattr_list(ino as u32)? {
            buf.extend_from_slice(name.as_bytes());
            buf.push(0);
        }
        if size != 0 && buf.len() > size as usize {
            return_errno!(Errno::ERANGE);
        }
        Ok(buf)
    }

    /// Remove an extended attribute.
    pub fn fuse_removexattr(&mut self, ino: u64, name: &str) -> Result<usize> {
        self.xattr_remove(ino as u32, name)?;
        Ok(EOK)
    }

    /// Test for a POSIX file lock.
    ///
    /// Returns a `FileLock` describing the first lock that *would* conflict with
    /// the requested `[start, end]` range of type `typ`; if nothing conflicts,
    /// the returned lock has type `F_UNLCK` (the range is grantable). `end` is
    /// inclusive, matching the table's convention.
    pub fn fuse_getlk(
        &self,
        ino: u64,
        _fh: u64,
        lock_owner: u64,
        start: u64,
        end: u64,
        typ: i32,
        pid: u32,
    ) -> Result<FileLock> {
        match self.lock_conflict(ino as u32, lock_owner, start, end, typ) {
            Some(conflict) => Ok(conflict),
            None => Ok(FileLock {
                owner: lock_owner,
                pid,
                start,
                end,
                typ: F_UNLCK,
            }),
        }
    }

    /// Acquire, modify or release a POSIX file lock.
    /// For POSIX threads (NPTL) there's a 1-1 relation between pid and owner, but
    /// otherwise this is not always the case.  For checking lock ownership,
    /// 'fi->owner' must be used. The l_pid field in 'struct flock' should only be
    /// used to fill in this field in getlk(). Note: if the locking methods are not
    /// implemented, the kernel will still allow file locking to work locally.
    /// Hence these are only interesting for network filesystems and similar.
    ///
    /// We can't truly block inside this synchronous library, so a blocking
    /// request (`sleep == true`) that conflicts returns `EAGAIN` just like the
    /// non-blocking case rather than waiting.
    pub fn fuse_setlk(
        &mut self,
        ino: u64,
        _fh: u64,
        lock_owner: u64,
        start: u64,
        end: u64,
        typ: i32,
        pid: u32,
        _sleep: bool,
    ) -> Result<usize> {
        let ino = ino as u32;

        if typ == F_UNLCK {
            self.lock_remove_owner_range(ino, lock_owner, start, end);
            return Ok(EOK);
        }

        // A read/write request must not clash with another owner's lock.
        if self
            .lock_conflict(ino, lock_owner, start, end, typ)
            .is_some()
        {
            return_errno!(Errno::EAGAIN);
        }

        // Drop this owner's own overlapping locks (upgrade/replace), then add it.
        self.lock_remove_owner_range(ino, lock_owner, start, end);
        self.locks.entry(ino).or_default().push(FileLock {
            owner: lock_owner,
            pid,
            start,
            end,
            typ,
        });
        Ok(EOK)
    }

    /// Find a held lock that conflicts with `[start, end]` of type `typ` for
    /// `owner`: a different owner, an overlapping range, and not both read locks.
    fn lock_conflict(
        &self,
        ino: u32,
        owner: u64,
        start: u64,
        end: u64,
        typ: i32,
    ) -> Option<FileLock> {
        let held = self.locks.get(&ino)?;
        held.iter()
            .find(|l| {
                l.owner != owner
                    && start <= l.end
                    && l.start <= end
                    && (typ == F_WRLCK || l.typ == F_WRLCK)
            })
            .copied()
    }

    /// Remove `owner`'s locks overlapping `[start, end]` for `ino`.
    fn lock_remove_owner_range(&mut self, ino: u32, owner: u64, start: u64, end: u64) {
        if let Some(held) = self.locks.get_mut(&ino) {
            held.retain(|l| !(l.owner == owner && start <= l.end && l.start <= end));
            if held.is_empty() {
                self.locks.remove(&ino);
            }
        }
    }

    /// Map block index within file to block index within device.
    /// Note: This makes sense only for block device backed filesystems mounted
    /// with the 'blkdev' option
    pub fn fuse_bmap(&self, ino: u64, blocksize: u32, idx: u64) -> Result<u64> {
        if blocksize == 0 {
            return_errno!(Errno::EINVAL);
        }
        let fs_bs = self.block_size() as u64;
        let bs = blocksize as u64;

        // `idx` counts the caller's `blocksize` units; translate to a byte
        // offset and then to the fs logical block that holds it.
        let byte_off = idx * bs;
        let lblock = (byte_off / fs_bs) as u32;

        let iref = self.get_inode_ref(ino as u32);
        let (pblock, _unwritten) = self.get_pblock_state(&iref, lblock);
        if pblock == 0 {
            return Ok(0); // hole / unmapped -> 0 by convention
        }

        // Physical byte offset, reported back in the caller's blocksize units.
        let phys_byte = pblock * fs_bs + (byte_off % fs_bs);
        Ok(phys_byte / bs)
    }

    /// control device
    ///
    /// Implements the ext4 inode-attribute ioctls a tool like `chattr`/`lsattr`
    /// or `stat` uses; anything else is rejected with `ENOTTY` (the kernel's
    /// "inappropriate ioctl for device"). The reply is the little-endian
    /// out-buffer FUSE hands back to the caller.
    pub fn fuse_ioctl(
        &self,
        ino: u64,
        _fh: u64,
        _flags: u32,
        cmd: u32,
        in_data: &[u8],
        _out_size: u32,
    ) -> Result<Vec<u8>> {
        // _IO numbers for the inode-flag / version ioctls (8-byte argument on
        // LP64, which is what the kernel and FUSE pass through).
        const FS_IOC_GETFLAGS: u32 = 0x8008_6601;
        const FS_IOC_SETFLAGS: u32 = 0x4008_6602;
        const FS_IOC_GETVERSION: u32 = 0x8008_7601;
        // Which i_flags bits a user may see (matches fs/ext4/ext4.h).
        const EXT4_FL_USER_VISIBLE: u32 = 0x705B_DFFF;
        // Bits SETFLAGS may change: the chattr-style attribute flags only. We
        // deliberately exclude on-disk-format flags the kernel also refuses to
        // toggle here (EXTENTS_FL 0x80000, HUGE_FILE_FL 0x40000, INDEX_FL
        // 0x1000, INLINE_DATA_FL, …) — clearing EXTENTS_FL on an extent inode
        // would corrupt it. So they are always preserved.
        const EXT4_FL_USER_MODIFIABLE: u32 = 0x6003_C0FF;

        match cmd {
            FS_IOC_GETFLAGS => {
                let iref = self.get_inode_ref(ino as u32);
                let visible = iref.inode.flags() & EXT4_FL_USER_VISIBLE;
                Ok(visible.to_le_bytes().to_vec())
            }
            FS_IOC_SETFLAGS => {
                if in_data.len() < 4 {
                    return_errno!(Errno::EINVAL);
                }
                let want = u32::from_le_bytes(in_data[..4].try_into().unwrap());
                let mut iref = self.get_inode_ref(ino as u32);
                // Replace only the user-modifiable bits; leave the rest
                // (EXTENTS_FL, INLINE_DATA_FL, …) untouched.
                let merged = (iref.inode.flags() & !EXT4_FL_USER_MODIFIABLE)
                    | (want & EXT4_FL_USER_MODIFIABLE);
                iref.inode.set_flags(merged);
                self.write_back_inode(&mut iref);
                Ok(Vec::new())
            }
            FS_IOC_GETVERSION => {
                let iref = self.get_inode_ref(ino as u32);
                Ok(iref.inode.generation().to_le_bytes().to_vec())
            }
            _ => return_errno!(Errno::ENOTTY),
        }
    }

    /// Poll for events.
    ///
    /// `events` is the mask the caller is waiting on; the reply is the subset
    /// that is currently ready (`revents`). A regular file on an on-disk
    /// filesystem never blocks — it is always readable and writable — so we
    /// report the requested read/write readiness immediately and never register
    /// the `kh` poll handle for later notification. Out-of-band / priority
    /// events (`POLLPRI` and friends) are never signalled.
    pub fn fuse_poll(
        &self,
        _ino: u64,
        _fh: u64,
        _kh: u64,
        events: u32,
        _flags: u32,
    ) -> Result<u32> {
        // Standard poll readiness bits for a never-blocking file.
        const POLLIN: u32 = 0x001;
        const POLLOUT: u32 = 0x004;
        const POLLRDNORM: u32 = 0x040;
        const POLLWRNORM: u32 = 0x100;
        const READY: u32 = POLLIN | POLLOUT | POLLRDNORM | POLLWRNORM;

        Ok(events & READY)
    }

    /// Preallocate or deallocate space to a file.
    ///
    /// Supports mode 0 (allocate and extend), `FALLOC_FL_KEEP_SIZE` (allocate
    /// without changing the size), and `FALLOC_FL_PUNCH_HOLE` (deallocate a
    /// range, leaving a hole). Other modes (COLLAPSE/ZERO/INSERT_RANGE) return
    /// `ENOTSUP`. The crate has no unwritten-extent support, so a preallocated
    /// hole is backed by real, zeroed blocks.
    pub fn fuse_fallocate(
        &self,
        ino: u64,
        fh: u64,
        offset: i64,
        length: i64,
        mode: i32,
    ) -> Result<()> {
        const FALLOC_FL_KEEP_SIZE: i32 = 0x01;
        const FALLOC_FL_PUNCH_HOLE: i32 = 0x02;
        const SUPPORTED: i32 = FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE;

        if offset < 0 || length <= 0 {
            return_errno_with_message!(Errno::EINVAL, "fallocate: bad offset/length");
        }
        if mode & !SUPPORTED != 0 {
            return_errno_with_message!(Errno::ENOTSUP, "fallocate: unsupported mode");
        }
        let punch = mode & FALLOC_FL_PUNCH_HOLE != 0;
        let keep_size = mode & FALLOC_FL_KEEP_SIZE != 0;
        // Linux requires PUNCH_HOLE to be combined with KEEP_SIZE.
        if punch && !keep_size {
            return_errno_with_message!(Errno::EINVAL, "fallocate: PUNCH_HOLE needs KEEP_SIZE");
        }

        let bs = self.block_size() as i64;
        let start = offset;
        let end = offset + length;
        let mut inode_ref = self.get_inode_ref(ino as u32);

        if punch {
            // Remove the extents of the fully-covered interior blocks; zero the
            // partial bytes in any mapped edge block (size never changes).
            let aligned_start = ((start + bs - 1) / bs) * bs; // round up
            let aligned_end = (end / bs) * bs; // round down
            if aligned_start < aligned_end {
                self.extent_remove_space(
                    &mut inode_ref,
                    (aligned_start / bs) as u32,
                    (aligned_end / bs - 1) as u32,
                )?;
                self.write_back_inode(&mut inode_ref);
                if start < aligned_start {
                    self.fallocate_zero_partial(&inode_ref, start, aligned_start);
                }
                if aligned_end < end {
                    self.fallocate_zero_partial(&inode_ref, aligned_end, end);
                }
            } else {
                // No whole block covered: a single contiguous partial span.
                self.fallocate_zero_partial(&inode_ref, start, end);
            }
            return Ok(());
        }

        // Allocate: fill only the holes in [start, end) with unwritten extents,
        // preserving existing data. Unwritten extents read back as zeros (the
        // read path honors the flag), so no disk zeroing is needed and e2fsck
        // does not require i_size to cover them.
        let start_blk = (start / bs) as usize;
        let end_blk = ((end + bs - 1) / bs) as usize; // exclusive
        let mut start_bgid = 0u32;
        let mut lb = start_blk;
        while lb < end_blk {
            if self.get_pblock_idx(&inode_ref, lb as u32).unwrap_or(0) != 0 {
                lb += 1;
                continue;
            }
            let mut run_end = lb + 1;
            while run_end < end_blk
                && self.get_pblock_idx(&inode_ref, run_end as u32).unwrap_or(0) == 0
            {
                run_end += 1;
            }
            let count = run_end - lb;
            let allocated =
                self.map_inode_pblk_batch(&mut inode_ref, &mut start_bgid, lb as u32, count, true)?;
            if allocated.len() < count {
                return_errno_with_message!(Errno::ENOSPC, "fallocate: out of space");
            }
            lb = run_end;
        }

        // Grow the size unless KEEP_SIZE was requested.
        if !keep_size && end as u64 > inode_ref.inode.size() {
            inode_ref.inode.set_size(end as u64);
            self.write_back_inode(&mut inode_ref);
        }
        Ok(())
    }

    /// Zero the byte range `[start, end)` within whatever mapped blocks it
    /// touches (a hole is already zeros, so unmapped blocks are skipped — they
    /// are not allocated). Used by PUNCH_HOLE for partial edge blocks.
    fn fallocate_zero_partial(&self, inode_ref: &Ext4InodeRef, start: i64, end: i64) {
        let bs = self.block_size() as i64;
        let mut lb = start / bs;
        while lb * bs < end {
            let blk_start = lb * bs;
            let lo = core::cmp::max(start, blk_start) - blk_start;
            let hi = core::cmp::min(end, blk_start + bs) - blk_start;
            let pblk = self.get_pblock_idx(inode_ref, lb as u32).unwrap_or(0);
            if pblk != 0 {
                let off = pblk as usize * bs as usize;
                let mut data = self.block_device.read_offset(off, bs as usize);
                for b in &mut data[lo as usize..hi as usize] {
                    *b = 0;
                }
                self.block_device.write_offset(off, &data);
            }
            lb += 1;
        }
    }

    /// Reposition read/write file offset.
    ///
    /// FUSE only forwards `SEEK_DATA`/`SEEK_HOLE` here (`SEEK_SET/CUR/END` are
    /// resolved by the kernel), so this locates data/hole boundaries in a sparse
    /// file at block granularity. Returns the resulting offset.
    pub fn fuse_lseek(&self, ino: u64, fh: u64, offset: i64, whence: i32) -> Result<i64> {
        const SEEK_DATA: i32 = 3;
        const SEEK_HOLE: i32 = 4;

        if whence != SEEK_DATA && whence != SEEK_HOLE {
            return_errno_with_message!(Errno::EINVAL, "lseek: unsupported whence");
        }
        if offset < 0 {
            return_errno_with_message!(Errno::EINVAL, "lseek: negative offset");
        }

        let inode_ref = self.get_inode_ref(ino as u32);
        let size = inode_ref.inode.size() as i64;
        // At or past EOF there is neither data nor an addressable hole.
        if offset >= size {
            return_errno_with_message!(Errno::ENXIO, "lseek: offset at or past EOF");
        }

        let bs = self.block_size() as i64;
        // A logical block with no extent maps to physical block 0 — a hole
        // (the same test read_at uses to zero-fill).
        let is_hole = |lblk: u32| -> bool {
            self.get_pblock_idx(&inode_ref, lblk)
                .map(|p| p == 0)
                .unwrap_or(true)
        };

        let want_data = whence == SEEK_DATA;
        let mut o = offset;
        while o < size {
            let lblk = (o / bs) as u32;
            if is_hole(lblk) != want_data {
                // SEEK_DATA wants a mapped block; SEEK_HOLE wants a hole.
                return Ok(o);
            }
            // Advance to the start of the next block.
            o = (lblk as i64 + 1) * bs;
        }

        if want_data {
            // Only holes remained before EOF.
            return_errno_with_message!(Errno::ENXIO, "lseek: no data after offset")
        } else {
            // The region from the last data block to EOF is an implicit hole.
            Ok(size)
        }
    }

    /// Copy the specified range from the source inode to the destination inode
    pub fn fuse_copy_file_range(
        &self,
        ino_in: u64,
        fh_in: u64,
        offset_in: i64,
        ino_out: u64,
        fh_out: u64,
        offset_out: i64,
        len: u64,
        flags: u32,
    ) -> Result<usize> {
        if flags != 0 {
            return_errno_with_message!(Errno::EINVAL, "copy_file_range: no flags supported");
        }
        if offset_in < 0 || offset_out < 0 {
            return_errno_with_message!(Errno::EINVAL, "copy_file_range: negative offset");
        }
        // Within one file, the source and destination ranges must not overlap.
        if ino_in == ino_out {
            let in_end = offset_in as u64 + len;
            let out_end = offset_out as u64 + len;
            if (offset_in as u64) < out_end && (offset_out as u64) < in_end {
                return_errno_with_message!(Errno::EINVAL, "copy_file_range: overlapping ranges");
            }
        }

        let chunk = self.block_size();
        let mut copied = 0usize;
        let mut in_off = offset_in as usize;
        let mut out_off = offset_out as usize;
        let mut buf = vec![0u8; chunk];

        while copied < len as usize {
            let want = core::cmp::min(chunk, len as usize - copied);
            // read_at clamps to the source size, so a short read means EOF.
            let n = self.read_at(ino_in as u32, in_off, &mut buf[..want])?;
            if n == 0 {
                break;
            }
            let written = self.write_at(ino_out as u32, out_off, &buf[..n])?;
            copied += written;
            in_off += written;
            out_off += written;
            if written < n {
                // Destination could not take everything (e.g. ENOSPC handled by
                // a short write); stop rather than spin.
                break;
            }
        }

        Ok(copied)
    }
}
