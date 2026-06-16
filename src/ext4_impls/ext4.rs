use crate::prelude::*;
use crate::return_errno_with_message;
use crate::utils::*;

use crate::ext4_defs::*;

impl Ext4 {
    /// The filesystem block size in bytes, read from the superblock.
    pub fn block_size(&self) -> usize {
        self.super_block.block_size() as usize
    }

    /// 获取system zone缓存
    pub fn get_system_zone(&self) -> Vec<SystemZone> {
        let mut zones = Vec::new();
        let group_count = self.super_block.block_group_count();
        let inodes_per_group = self.super_block.inodes_per_group();
        let inode_size = self.super_block.inode_size() as u64;
        let block_size = self.super_block.block_size() as u64;
        for bgid in 0..group_count {
            // meta blocks
            let meta_blks = self.num_base_meta_blocks(bgid);
            if meta_blks != 0 {
                let start = self.get_block_of_bgid(bgid);
                zones.push(SystemZone {
                    group: bgid,
                    start_blk: start,
                    end_blk: start + meta_blks as u64 - 1,
                });
            }
            // block group描述符
            let block_group =
                Ext4BlockGroup::load_new(&self.block_device, &self.super_block, bgid as usize);
            // block bitmap
            let blk_bmp = block_group.get_block_bitmap_block(&self.super_block);
            zones.push(SystemZone {
                group: bgid,
                start_blk: blk_bmp,
                end_blk: blk_bmp,
            });
            // inode bitmap
            let ino_bmp = block_group.get_inode_bitmap_block(&self.super_block);
            zones.push(SystemZone {
                group: bgid,
                start_blk: ino_bmp,
                end_blk: ino_bmp,
            });
            // inode table
            let ino_tbl = block_group.get_inode_table_blk_num() as u64;
            let itb_per_group =
                ((inodes_per_group as u64 * inode_size + block_size - 1) / block_size) as u64;
            zones.push(SystemZone {
                group: bgid,
                start_blk: ino_tbl,
                end_blk: ino_tbl + itb_per_group - 1,
            });
        }
        zones
    }

    /// Read a fresh copy of the superblock from disk.
    ///
    /// The geometry fields are immutable, but the free inode/block counters
    /// change as allocations happen. `self.super_block` is captured at `open`
    /// and never updated (the allocation methods take `&self`), so a free-count
    /// read-modify-write that starts from it persists only a single decrement
    /// no matter how many allocations ran. Reading the live on-disk value
    /// immediately before each such update keeps the counters correct. The
    /// block-group descriptors are already handled this way (`load_new`).
    pub fn read_super_block(&self) -> Ext4Superblock {
        let block = Block::load(&self.block_device, SUPERBLOCK_OFFSET, SUPERBLOCK_OFFSET);
        block.read_as()
    }

    /// Opens and loads an Ext4 from the `block_device`.
    pub fn open(block_device: Arc<dyn BlockDevice>) -> Self {
        // Load the superblock. It lives at byte 1024 and is 1024 bytes; the
        // block size isn't known yet, so read a fixed amount.
        let block = Block::load(&block_device, SUPERBLOCK_OFFSET, SUPERBLOCK_OFFSET);
        let super_block: Ext4Superblock = block.read_as();

        // drop(block);

        let ext4_tmp = Ext4 {
            block_device,
            super_block,
            system_zone_cache: None,
            locks: BTreeMap::new(),
            journal_device: None,
            journal: None,
        };
        let zones = ext4_tmp.get_system_zone();

        Ext4 {
            system_zone_cache: Some(zones),
            ..ext4_tmp
        }
    }

    /// Open the filesystem and run journal recovery if a dirty journal is present.
    /// On a clean image (or one without a journal) this behaves like `open`.
    /// This is the entry point a mounting caller (FUSE) should use.
    pub fn open_and_recover(block_device: Arc<dyn BlockDevice>) -> Result<Self> {
        let fs = Ext4::open(block_device);
        if let Some(journal) = crate::ext4_impls::journal::Journal::load(&fs)? {
            journal.recover(&fs)?; // no-op when the journal is clean
        }
        Ok(fs)
    }

    /// Open the fs with journaling enabled: wrap the device in a JournalDevice,
    /// recover any dirty journal, and store the engine so mutating ops are journaled.
    /// Falls back to a plain (un-journaled) open if the fs has no journal.
    pub fn open_journaled(file_dev: Arc<dyn BlockDevice>) -> Result<Self> {
        use crate::ext4_impls::journal::{Journal, JournalDevice};
        let bs = Ext4::open(file_dev.clone()).block_size(); // probe block size
        let jdev = Arc::new(JournalDevice::new(file_dev, bs));
        let mut fs = Ext4::open(jdev.clone()); // block_device = jdev (inactive → passthrough)
        if let Some(journal) = Journal::load(&fs)? {
            journal.recover(&fs)?; // replay a dirty journal (jdev passthrough)
            fs.journal = Some(journal);
            fs.journal_device = Some(jdev);
        }
        Ok(fs)
    }

    /// Begin the running transaction for a mutating op (no-op if not journaled).
    pub fn journal_begin(&self) -> Result<()> {
        if let (Some(jd), Some(j)) = (&self.journal_device, &self.journal) {
            let seq = j.live_sequence(self)?;
            jd.begin(seq);
        }
        Ok(())
    }

    /// End a mutating op: commit the transaction if `ok`, else discard it (no-op if
    /// not journaled). Honors a test-injected crash point.
    ///
    /// On `ok == false` the running transaction is discarded — its captured writes
    /// never reach disk (atomic abort).
    pub fn journal_end(&self, ok: bool) -> Result<()> {
        if let (Some(jd), Some(j)) = (&self.journal_device, &self.journal) {
            if let Some(txn) = jd.end() {
                if ok {
                    let crash = jd.take_crash();
                    j.commit_with_crash(self, &txn, crash)?;
                }
                // if !ok: drop txn → captured writes never hit disk (atomic abort)
            }
        }
        Ok(())
    }

    /// Run a mutating filesystem operation inside a journal transaction: begin,
    /// run `op`, then commit if it succeeded or discard (atomic abort) if it
    /// failed. A no-op wrapper when journaling is off. This is the ONLY correct way
    /// to journal an op — it guarantees the transaction is always closed, so the
    /// body may use `?` freely without leaking an open transaction.
    pub fn journaled<T>(&self, op: impl FnOnce() -> Result<T>) -> Result<T> {
        self.journal_begin()?;
        let r = op();
        self.journal_end(r.is_ok())?;
        r
    }

    /// Leak-safe journaling wrapper for `&mut self` operations. Same contract as
    /// `journaled` (begin → op → commit-or-abort, no-op when not journaled), but the
    /// op receives `&mut Self`. Use this for fuse ops that need `&mut self`.
    pub fn journaled_mut<T>(&mut self, op: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.journal_begin()?;
        let r = op(self);
        self.journal_end(r.is_ok())?;
        r
    }

    // with dir result search path offset
    pub fn generic_open(
        &self,
        path: &str,
        parent_inode_num: &mut u32,
        create: bool,
        ftype: u16,
        name_off: &mut u32,
    ) -> Result<u32> {
        let mut is_goal = false;

        let mut parent = parent_inode_num;

        let mut search_path = path;

        let mut dir_search_result = Ext4DirSearchResult::new(Ext4DirEntry::default());

        loop {
            while search_path.starts_with('/') {
                *name_off += 1; // Skip the slash
                search_path = &search_path[1..];
            }

            let len = path_check(search_path, &mut is_goal);

            let current_path = &search_path[..len];

            if len == 0 || search_path.is_empty() {
                break;
            }

            search_path = &search_path[len..];

            let r = self.dir_find_entry(*parent, current_path, &mut dir_search_result);

            // log::trace!("find in parent {:x?} r {:?} name {:?}", parent, r, current_path);
            if let Err(e) = r {
                if e.error() != Errno::ENOENT || !create {
                    return_errno_with_message!(Errno::ENOENT, "No such file or directory");
                }

                let mut inode_mode = 0;
                if is_goal {
                    inode_mode = ftype;
                } else {
                    inode_mode = InodeFileType::S_IFDIR.bits();
                }

                let new_inode_ref = self.create(*parent, current_path, inode_mode)?;

                // Update parent to the new inode
                *parent = new_inode_ref.inode_num;

                // Now, update dir_search_result to reflect the new inode
                dir_search_result.dentry.inode = new_inode_ref.inode_num;

                continue;
            }

            if is_goal {
                break;
            } else {
                // update parent
                *parent = dir_search_result.dentry.inode;
            }
            *name_off += len as u32;
        }

        if is_goal {
            return Ok(dir_search_result.dentry.inode);
        }

        Ok(dir_search_result.dentry.inode)
    }

    #[allow(unused)]
    pub fn dir_mk(&self, path: &str) -> Result<usize> {
        let mut nameoff = 0;

        let filetype = InodeFileType::S_IFDIR;

        // todo get this path's parent

        // start from root
        let mut parent = ROOT_INODE;

        let r = self.generic_open(path, &mut parent, true, filetype.bits(), &mut nameoff);
        Ok(EOK)
    }

    pub fn unlink(
        &self,
        parent: &mut Ext4InodeRef,
        child: &mut Ext4InodeRef,
        name: &str,
    ) -> Result<usize> {
        self.journaled(|| self.unlink_impl(parent, child, name))
    }

    fn unlink_impl(
        &self,
        parent: &mut Ext4InodeRef,
        child: &mut Ext4InodeRef,
        name: &str,
    ) -> Result<usize> {
        self.dir_remove_entry(parent, name)?;

        // Drop the directory's link to the child and persist it. Without this
        // the on-disk inode still shows links_count > 0, so a consistency
        // checker sees a live but unreferenced inode even after the bitmap bit
        // is cleared.
        let links = child.inode.links_count().saturating_sub(1);
        child.inode.set_links_count(links);

        if links == 0 {
            // Last link gone: reset the slot to the all-zero unused state (what
            // mkfs leaves) before releasing it. A deleted inode that keeps a
            // non-zero i_mode with dtime == 0 reads as "deleted inode has zero
            // dtime"; a small non-zero dtime instead reads as an orphan-list
            // pointer. Zeroing sidesteps both (no wall clock in no_std).
            let is_dir = child.inode.is_dir();
            child.inode = Ext4Inode::default();
            self.write_back_inode_without_csum(child);
            self.ialloc_free_inode(child.inode_num, is_dir);
        } else {
            self.write_back_inode(child);
        }

        Ok(EOK)
    }
}
