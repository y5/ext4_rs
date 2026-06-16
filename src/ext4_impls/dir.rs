use crate::prelude::*;
use crate::return_errno_with_message;

use crate::ext4_defs::*;

/// Map an inode's format to the directory-entry type byte stored in its parent
/// directory entry (the `file_type` field, valid in ext4 rev >= 0.5).
fn de_type_from_inode(inode: &Ext4Inode) -> DirEntryType {
    match inode.file_type() {
        InodeFileType::S_IFREG => DirEntryType::EXT4_DE_REG_FILE,
        InodeFileType::S_IFDIR => DirEntryType::EXT4_DE_DIR,
        InodeFileType::S_IFLNK => DirEntryType::EXT4_DE_SYMLINK,
        InodeFileType::S_IFCHR => DirEntryType::EXT4_DE_CHRDEV,
        InodeFileType::S_IFBLK => DirEntryType::EXT4_DE_BLKDEV,
        InodeFileType::S_IFIFO => DirEntryType::EXT4_DE_FIFO,
        InodeFileType::S_IFSOCK => DirEntryType::EXT4_DE_SOCK,
        _ => DirEntryType::EXT4_DE_UNKNOWN,
    }
}

impl Ext4 {
    /// Find a directory entry in a directory
    ///
    /// Params:
    /// parent_inode: u32 - inode number of the parent directory
    /// name: &str - name of the entry to find
    /// result: &mut Ext4DirSearchResult - result of the search
    ///
    /// Returns:
    /// `Result<usize>` - status of the search
    pub fn dir_find_entry(
        &self,
        parent_inode: u32,
        name: &str,
        result: &mut Ext4DirSearchResult,
    ) -> Result<usize> {
        // load parent inode
        let parent = self.get_inode_ref(parent_inode);
        if !parent.inode.is_dir() {
            return_errno_with_message!(Errno::ENOTDIR, "dir_find_entry on non-directory inode");
        }

        // start from the first logical block
        let mut iblock = 0;
        // physical block id
        let mut fblock: Ext4Fsblk = 0;

        // calculate total blocks
        let inode_size: u64 = parent.inode.size();
        let total_blocks: u64 = inode_size / self.block_size() as u64;

        // iterate all blocks
        while iblock < total_blocks {
            let search_path = self.find_extent(&parent, iblock as u32);

            if let Ok(path) = search_path {
                // get the last path
                let path = path.path.last().unwrap();

                // get physical block id
                fblock = path.pblock;

                // load physical block
                let mut ext4block = Block::load(&self.block_device, fblock as usize * self.block_size(), self.block_size());

                // find entry in block
                let r = self.dir_find_in_block(&ext4block, name, result);

                if r.is_ok() {
                    result.pblock_id = fblock as usize;
                    return Ok(EOK);
                }
            } else {
                return_errno_with_message!(Errno::ENOENT, "dir search fail")
            }
            // go to next block
            iblock += 1
        }

        return_errno_with_message!(Errno::ENOENT, "dir search fail");
    }

    /// Find a directory entry in a block
    ///
    /// Params:
    /// block: &mut Block - block to search in
    /// name: &str - name of the entry to find
    ///
    /// Returns:
    /// result: Ext4DirEntry - result of the search
    pub fn dir_find_in_block(
        &self,
        block: &Block,
        name: &str,
        result: &mut Ext4DirSearchResult,
    ) -> Result<Ext4DirEntry> {
        let mut offset = 0;
        let mut prev_de_offset = 0;

        // start from the first entry
        while offset < self.block_size() - core::mem::size_of::<Ext4DirEntryTail>() {
            let de: Ext4DirEntry = block.read_offset_as(offset);
            if !de.unused() && de.compare_name(name) {
                result.dentry = de;
                result.offset = offset;
                result.prev_offset = prev_de_offset;
                return Ok(de);
            }

            prev_de_offset = offset;
            // go to next entry; guard against a malformed zero-length entry
            // that would otherwise spin forever.
            let de_len = de.entry_len() as usize;
            if de_len == 0 {
                break;
            }
            offset += de_len;
        }
        return_errno_with_message!(Errno::ENOENT, "dir find in block failed");
    }

    /// Get dir entries of a inode
    ///
    /// Params:
    /// inode: u32 - inode number of the directory
    ///
    /// Returns:
    /// `Vec<Ext4DirEntry>` - list of directory entries
    pub fn dir_get_entries(&self, inode: u32) -> Vec<Ext4DirEntry> {
        self.dir_entries_with_offset_from(inode, 0)
            .into_iter()
            .map(|e| e.entry)
            .collect()
    }

    /// Walk a directory's entries starting at the byte-position cookie `start`,
    /// pairing each used entry with the resume cookie that follows it.
    ///
    /// The cookie of an entry is its absolute byte position within the
    /// directory file (`iblock * block_size + offset_in_block`); `next_offset`
    /// is the position of the following entry. Entries whose position is below
    /// `start` are skipped, so a caller can resume an enumeration mid-stream by
    /// passing the last cookie it saw. Unused slots and the directory tail are
    /// never emitted.
    pub fn dir_entries_with_offset_from(
        &self,
        inode: u32,
        start: u64,
    ) -> Vec<Ext4DirEntryWithOffset> {
        let mut entries = Vec::new();

        // load inode
        let inode_ref = self.get_inode_ref(inode);
        // We return an empty Vec rather than panicking when the caller
        // hands us a non-directory inode
        if !inode_ref.inode.is_dir() {
            return entries;
        }

        let block_size = self.block_size();

        // calculate total blocks
        let inode_size = inode_ref.inode.size();
        let total_blocks = inode_size / block_size as u64;

        // start from the first logical block
        let mut iblock = 0;

        // iterate all blocks
        while iblock < total_blocks {
            // get physical block id of a logical block id
            let search_path = self.find_extent(&inode_ref, iblock as u32);

            if let Ok(path) = search_path {
                // get the last path
                let path = path.path.last().unwrap();

                // get physical block id
                let fblock = path.pblock;

                // load physical block
                let ext4block = Block::load(&self.block_device, fblock as usize * block_size, block_size);
                let mut offset = 0;

                // iterate all entries in a block
                while offset < block_size - core::mem::size_of::<Ext4DirEntryTail>() {
                    let de: Ext4DirEntry = ext4block.read_offset_as(offset);
                    // absolute byte position of this entry within the directory
                    let pos = iblock * block_size as u64 + offset as u64;
                    // guard against a malformed zero-length entry (infinite loop)
                    let de_len = de.entry_len() as usize;
                    if de_len == 0 {
                        break;
                    }
                    if !de.unused() && pos >= start {
                        entries.push(Ext4DirEntryWithOffset {
                            entry: de,
                            next_offset: pos + de_len as u64,
                        });
                    }
                    offset += de_len;
                }
            }

            // go ot next block
            iblock += 1;
        }
        entries
    }

    /// Recompute and store a directory leaf block's tail checksum. `dir_ino` is
    /// the inode number of the directory that owns the block — it seeds the
    /// checksum and must be passed explicitly, since an appended block's first
    /// entry is a regular entry rather than "." (see `ext4_dir_block_csum`).
    pub fn dir_set_csum(&self, dst_blk: &mut Block, dir_ino: u32, ino_gen: u32) {
        let tail_offset = self.block_size() - size_of::<Ext4DirEntryTail>();
        let mut tail: Ext4DirEntryTail = *dst_blk.read_offset_as_mut(tail_offset);

        tail.tail_set_csum(&self.super_block, dir_ino, ino_gen, &dst_blk.data[..]);

        tail.copy_to_slice(&mut dst_blk.data);
    }

    /// Add a new entry to a directory
    ///
    /// Params:
    /// parent: &mut Ext4InodeRef - parent directory inode reference
    /// child: &mut Ext4InodeRef - child inode reference
    /// path: &str - path of the new entry
    ///
    /// Returns:
    /// `Result<usize>` - status of the operation
    pub fn dir_add_entry(
        &self,
        parent: &mut Ext4InodeRef,
        child: &Ext4InodeRef,
        name: &str,
    ) -> Result<usize> {
        // Directory entries carry the referenced inode's type. Derive it from
        // the child inode instead of assuming a directory - hardcoding
        // EXT4_DE_DIR made every newly created regular file show up as a
        // directory (e.g. `ls` reporting `d` for a plain file).
        let de_type = de_type_from_inode(&child.inode);

        // calculate total blocks
        let inode_size: u64 = parent.inode.size();
        let block_size = self.super_block.block_size();
        let total_blocks: u64 = inode_size / block_size as u64;

        // iterate all blocks
        let mut iblock = 0;
        while iblock < total_blocks {
            // get physical block id of a logical block id
            let pblock = self.get_pblock_idx(parent, iblock as u32)?;

            // load physical block
            let mut ext4block = Block::load(&self.block_device, pblock as usize * self.block_size(), self.block_size());

            let result =
                self.try_insert_to_existing_block(&mut ext4block, name, child.inode_num, de_type);

            if result.is_ok() {
                // set checksum
                self.dir_set_csum(&mut ext4block, parent.inode_num, parent.inode.generation());
                ext4block.sync_blk_to_disk(&self.block_device);

                return Ok(EOK);
            }

            // go ot next block
            iblock += 1;
        }

        // no space in existing blocks, need to add new block
        let new_block = self.append_inode_pblk(parent)?;

        // load new block
        let mut new_ext4block = Block::load(&self.block_device, new_block as usize * self.block_size(), self.block_size());

        // write new entry to the new block
        // must succeed, as we just allocated the block
        self.insert_to_new_block(&mut new_ext4block, child.inode_num, name, de_type);

        // set checksum
        self.dir_set_csum(&mut new_ext4block, parent.inode_num, parent.inode.generation());
        new_ext4block.sync_blk_to_disk(&self.block_device);

        Ok(EOK)
    }

    /// Try to insert a new entry to an existing block
    ///
    /// Params:
    /// block: &mut Block - block to insert the new entry
    /// name: &str - name of the new entry
    /// inode: u32 - inode number of the new entry
    ///
    /// Returns:
    /// `Result<usize>` - status of the operation
    pub fn try_insert_to_existing_block(
        &self,
        block: &mut Block,
        name: &str,
        child_inode: u32,
        de_type: DirEntryType,
    ) -> Result<usize> {
        // required length aligned to 4 bytes
        let required_len = {
            let mut len = size_of::<Ext4DirEntry>() + name.len();
            if len % 4 != 0 {
                len += 4 - (len % 4);
            }
            len
        };

        let mut offset = 0;

        // Start from the first entry
        while offset < self.block_size() - size_of::<Ext4DirEntryTail>() {
            let mut de = Ext4DirEntry::try_from(&block.data[offset..]).unwrap();

            // Record length used to advance to the next entry.
            // A zero rec_len (corrupt/malformed directory block) would leave `offset`
            // unchanged and spin this loop forever.
            // Bail out instead of hanging.
            let entry_rec_len = de.entry_len() as usize;
            if entry_rec_len == 0 {
                break;
            }

            if de.unused() {
                // We skip the free/unused slot. `offset` MUST advance here
                // since a bare `continue` would re-read the same entry forever.
                // This would cause an infinite loop on any directory block containing
                // an unused entry.
                offset += entry_rec_len;
                continue;
            }

            let inode = de.inode;
            let rec_len = de.entry_len;

            let used_len = de.name_len as usize;
            let mut sz = core::mem::size_of::<Ext4FakeDirEntry>() + used_len;
            if used_len % 4 != 0 {
                sz += 4 - used_len % 4;
            }

            let free_space = rec_len as usize - sz;

            // If there is enough free space
            if free_space >= required_len {
                // Create new directory entry
                let mut new_entry = Ext4DirEntry::default();

                // Update existing entry length and copy both entries back to block data
                de.entry_len = sz as u16;

                new_entry.write_entry(free_space as u16, child_inode, name, de_type);

                // update parent_de and new_de to blk_data
                de.copy_to_slice(&mut block.data, offset);
                new_entry.copy_to_slice(&mut block.data, offset + sz);

                // Sync to disk
                block.sync_blk_to_disk(&self.block_device);

                return Ok(EOK);
            }

            // Move to the next entry
            offset += entry_rec_len;
        }

        return_errno_with_message!(Errno::ENOSPC, "No space in block for new entry");
    }

    /// Insert a new entry to a new block
    ///
    /// Params:
    /// block: &mut Block - block to insert the new entry
    /// name: &str - name of the new entry
    /// inode: u32 - inode number of the new entry
    pub fn insert_to_new_block(
        &self,
        block: &mut Block,
        inode: u32,
        name: &str,
        de_type: DirEntryType,
    ) {
        // write new entry
        let mut new_entry = Ext4DirEntry::default();
        let el = self.block_size() - size_of::<Ext4DirEntryTail>();
        new_entry.write_entry(el as u16, inode, name, de_type);
        new_entry.copy_to_slice(&mut block.data, 0);

        copy_dir_entry_to_array(&new_entry, &mut block.data, 0);

        // init tail for new block
        let tail = Ext4DirEntryTail::new();
        tail.copy_to_slice(&mut block.data);
    }

    pub fn dir_remove_entry(&self, parent: &mut Ext4InodeRef, path: &str) -> Result<usize> {
        // get remove_entry pos in parent and its prev entry
        let mut result = Ext4DirSearchResult::new(Ext4DirEntry::default());

        let r = self.dir_find_entry(parent.inode_num, path, &mut result)?;

        let mut ext4block = Block::load(&self.block_device, result.pblock_id * self.block_size(), self.block_size());

        // Invalidate entry first
        let de_del: &mut Ext4DirEntry = ext4block.read_offset_as_mut(result.offset);
        de_del.inode = 0;

        // Store entry position in block
        let pos = result.offset;

        // If entry is not the first in block, it must be merged with previous entry
        if pos != 0 {
            let mut offset = 0;

            // Start from the first entry in block
            let mut tmp_de: Ext4DirEntry = ext4block.read_offset_as(offset);
            let mut de_len = tmp_de.entry_len();

            // Find direct predecessor of removed entry
            while (offset + de_len as usize) < pos {
                // A zero-length entry would never advance `offset`; bail out
                // instead of looping forever on a malformed block.
                if de_len == 0 {
                    return_errno_with_message!(Errno::EINVAL, "dir remove: zero-length entry");
                }
                offset += de_len as usize;
                tmp_de = ext4block.read_offset_as(offset);
                de_len = tmp_de.entry_len();
            }

            if de_len as usize + offset != pos {
                return_errno_with_message!(Errno::EINVAL, "Invalid predecessor calculation");
            }

            // Add removed entry length to predecessor's length
            let del_len = result.dentry.entry_len();
            let mut tmp_de_mut: &mut Ext4DirEntry = ext4block.read_offset_as_mut(offset);
            tmp_de_mut.entry_len = de_len + del_len;
        }

        self.dir_set_csum(&mut ext4block, parent.inode_num, parent.inode.generation());
        ext4block.sync_blk_to_disk(&self.block_device);

        Ok(EOK)
    }

    /// Repoint an existing directory entry at a different inode, updating the
    /// stored file-type byte to match. The entry keeps its name and position;
    /// only its target changes. Used by rename to retarget a '..' back-reference
    /// and to swap two names with RENAME_EXCHANGE.
    pub fn dir_set_entry_target(
        &self,
        parent_ino: u32,
        name: &str,
        target_ino: u32,
    ) -> Result<usize> {
        let mut res = Ext4DirSearchResult::new(Ext4DirEntry::default());
        self.dir_find_entry(parent_ino, name, &mut res)?;

        let de_type = de_type_from_inode(&self.get_inode_ref(target_ino).inode);

        let mut blk = Block::load(
            &self.block_device,
            res.pblock_id * self.block_size(),
            self.block_size(),
        );
        let de: &mut Ext4DirEntry = blk.read_offset_as_mut(res.offset);
        de.inode = target_ino;
        de.inner.inode_type = de_type.bits();

        let gen = self.get_inode_ref(parent_ino).inode.generation();
        self.dir_set_csum(&mut blk, parent_ino, gen);
        blk.sync_blk_to_disk(&self.block_device);

        Ok(EOK)
    }

    pub fn dir_has_entry(&self, dir_inode: u32) -> bool {
        // load parent inode
        let parent = self.get_inode_ref(dir_inode);
        // Non-directories have no entries
        if !parent.inode.is_dir() {
            return false;
        }

        // start from the first logical block
        let mut iblock = 0;
        // physical block id
        let mut fblock: Ext4Fsblk = 0;

        // calculate total blocks
        let inode_size: u64 = parent.inode.size();
        let total_blocks: u64 = inode_size / self.block_size() as u64;

        // iterate all blocks
        while iblock < total_blocks {
            let search_path = self.find_extent(&parent, iblock as u32);

            if let Ok(path) = search_path {
                // get the last path
                let path = path.path.last().unwrap();

                // get physical block id
                fblock = path.pblock;

                // load physical block
                let ext4block = Block::load(&self.block_device, fblock as usize * self.block_size(), self.block_size());

                // start from the first entry
                let mut offset = 0;
                while offset < self.block_size() - core::mem::size_of::<Ext4DirEntryTail>() {
                    let de: Ext4DirEntry = ext4block.read_offset_as(offset);
                    // guard against a malformed zero-length entry (infinite loop)
                    let de_len = de.entry_len as usize;
                    if de_len == 0 {
                        break;
                    }
                    offset += de_len;
                    if de.inode == 0 {
                        continue;
                    }
                    // skip . and ..
                    if de.get_name() == "." || de.get_name() == ".." {
                        continue;
                    }
                    return true;
                }
            }
            // go to next block
            iblock += 1
        }

        false
    }

    pub fn dir_remove(&self, parent: u32, path: &str) -> Result<usize> {
        let mut search_result = Ext4DirSearchResult::new(Ext4DirEntry::default());

        let r = self.dir_find_entry(parent, path, &mut search_result)?;

        let mut parent_inode_ref = self.get_inode_ref(parent);
        let mut child_inode_ref = self.get_inode_ref(search_result.dentry.inode);

        // Only empty directories (containing just '.' and '..') can be removed.
        if self.dir_has_entry(child_inode_ref.inode_num) {
            return_errno_with_message!(Errno::ENOTEMPTY, "directory not empty")
        }

        // Remove the entry from the parent directory.
        self.dir_remove_entry(&mut parent_inode_ref, path)?;

        // Free the directory's data block (the '.'/'..' block).
        self.truncate_inode(&mut child_inode_ref, 0)?;

        // A removed empty directory loses both the parent's entry and its own
        // '.' self-link, so its link count drops to zero. Reset the inode to the
        // unused state and release it from the inode bitmap.
        child_inode_ref.inode = Ext4Inode::default();
        self.write_back_inode_without_csum(&child_inode_ref);
        self.ialloc_free_inode(child_inode_ref.inode_num, true);

        // The parent loses the '..' back-reference from the removed subdirectory.
        let parent_links = parent_inode_ref.inode.links_count().saturating_sub(1);
        parent_inode_ref.inode.set_links_count(parent_links);
        self.write_back_inode(&mut parent_inode_ref);

        Ok(EOK)
    }
}

pub fn copy_dir_entry_to_array(header: &Ext4DirEntry, array: &mut [u8], offset: usize) {
    unsafe {
        let de_ptr = header as *const Ext4DirEntry as *const u8;
        let array_ptr = array as *mut [u8] as *mut u8;
        let count = core::mem::size_of::<Ext4DirEntry>() / core::mem::size_of::<u8>();
        core::ptr::copy_nonoverlapping(de_ptr, array_ptr.add(offset), count);
    }
}
