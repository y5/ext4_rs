//! Root-free ext4 fixture harness.
//!
//! Each test builds a real ext4 image with `mkfs.ext4` at a chosen block size,
//! populates it with `debugfs` (no mount, no root), then drives the crate
//! against it and checks the result with `e2fsck -fn`.
//!
//! The 4 KiB cases are the baseline: they must pass on the current code, which
//! proves the harness itself is correct. The 1 KiB cases are the target for the
//! dynamic-block-size work: they fail until the crate stops assuming 4 KiB.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use ext4_rs::{BlockDevice, Errno, Ext4, Ext4DirEntry, Ext4DirSearchResult, InodeFileType};

const ROOT_INODE: u32 = 2;

/// A file-backed block device over an on-disk image.
struct FileBlockDevice {
    path: PathBuf,
}

impl FileBlockDevice {
    fn new(path: impl Into<PathBuf>) -> Self {
        FileBlockDevice { path: path.into() }
    }
}

impl BlockDevice for FileBlockDevice {
    fn read_offset(&self, offset: usize, len: usize) -> Vec<u8> {
        use std::io::{Read, Seek};
        let mut buf = vec![0u8; len];
        let mut file = fs::OpenOptions::new().read(true).open(&self.path).unwrap();
        file.seek(std::io::SeekFrom::Start(offset as u64)).unwrap();
        // Allow short reads at end-of-device.
        let mut filled = 0;
        while filled < buf.len() {
            match file.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) => panic!("read_offset: {e}"),
            }
        }
        buf
    }

    fn write_offset(&self, offset: usize, data: &[u8]) {
        use std::io::{Seek, Write};
        let mut file = fs::OpenOptions::new().write(true).open(&self.path).unwrap();
        file.seek(std::io::SeekFrom::Start(offset as u64)).unwrap();
        file.write_all(data).unwrap();
    }
}

/// Deterministic, non-repeating-enough payload so a wrong-offset read can't
/// accidentally match. `len` bytes.
fn payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 31 + 7) % 251) as u8).collect()
}

fn tool_missing(name: &str) -> bool {
    Command::new(name)
        .arg("-V")
        .output()
        .map(|_| false)
        .unwrap_or(true)
}

/// Create a fresh, empty ext4 image at `block_size`. `tag` keeps parallel
/// tests from sharing a path.
fn fresh_image(block_size: u32, tag: &str) -> PathBuf {
    let dir = Path::new("target").join("harness");
    fs::create_dir_all(&dir).unwrap();

    let img = dir.join(format!("{}_{}.img", tag, block_size));
    let _ = fs::remove_file(&img);

    // 16 MiB image is plenty for these fixtures.
    let zeros = vec![0u8; 16 * 1024 * 1024];
    fs::write(&img, &zeros).unwrap();

    let status = Command::new("mkfs.ext4")
        .args(["-q", "-b", &block_size.to_string(), "-F"])
        .arg(&img)
        .status()
        .expect("mkfs.ext4 failed to spawn");
    assert!(status.success(), "mkfs.ext4 -b {block_size} failed");
    img
}

/// Build a fresh image at `block_size`, with a single known file written via
/// debugfs at `/probe.bin`. Returns the image path and the expected contents.
fn make_image(block_size: u32, probe_len: usize) -> (PathBuf, Vec<u8>) {
    let dir = Path::new("target").join("harness");
    let img = fresh_image(block_size, "fixture");
    let probe = dir.join(format!("probe_{}.bin", block_size));

    let data = payload(probe_len);
    fs::write(&probe, &data).unwrap();

    let status = Command::new("debugfs")
        .arg("-w")
        .arg("-R")
        .arg(format!("write {} /probe.bin", probe.display()))
        .arg(&img)
        .status()
        .expect("debugfs failed to spawn");
    assert!(status.success(), "debugfs write failed");

    (img, data)
}

fn fsck_clean(img: &Path) {
    let out = Command::new("e2fsck")
        .args(["-fn"])
        .arg(img)
        .output()
        .expect("e2fsck failed to spawn");
    assert!(
        out.status.success(),
        "e2fsck -fn reported errors:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Full e2fsck report (stdout+stderr) regardless of exit status. Needed for
/// discrepancies e2fsck reports but does not treat as fatal (e.g. superblock
/// free-count summaries), which `fsck_clean` would not catch.
fn fsck_output(img: &Path) -> String {
    let out = Command::new("e2fsck")
        .args(["-fn"])
        .arg(img)
        .output()
        .expect("e2fsck failed to spawn");
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Run a debugfs command against the image (read-write), returning stdout.
fn debugfs(img: &Path, request: &str) -> String {
    let out = Command::new("debugfs")
        .arg("-w")
        .arg("-R")
        .arg(request)
        .arg(img)
        .output()
        .expect("debugfs failed to spawn");
    assert!(out.status.success(), "debugfs `{request}` failed");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Open the image with the crate, read `/probe.bin`, and assert the bytes match
/// what debugfs wrote. This is the block-size-independent contract.
fn read_probe_matches(block_size: u32) {
    if tool_missing("mkfs.ext4") || tool_missing("debugfs") || tool_missing("e2fsck") {
        eprintln!("skipping: e2fsprogs tooling not available");
        return;
    }

    let probe_len = 10_000; // spans several blocks at every block size
    let (img, expected) = make_image(block_size, probe_len);

    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let ext4 = Ext4::open(dev);

    let inode = ext4
        .generic_open("/probe.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("could not resolve /probe.bin");

    let mut buf = vec![0u8; probe_len];
    let n = ext4
        .read_at(inode, 0, &mut buf)
        .expect("read_at /probe.bin failed");

    assert_eq!(n, probe_len, "short read at block size {block_size}");
    assert_eq!(
        buf, expected,
        "content mismatch at block size {block_size} (wrong block-offset math?)"
    );

    // Reading must not have mutated anything.
    fsck_clean(&img);
}

#[test]
fn read_probe_4k_baseline() {
    read_probe_matches(4096);
}

#[test]
fn read_probe_2k() {
    read_probe_matches(2048);
}

#[test]
fn read_probe_1k_target() {
    read_probe_matches(1024);
}

/// Create a file with the crate, write a multi-block payload, require e2fsck to
/// be clean, then reopen and read it back. Exercises the write path: inode/block
/// allocation, directory insertion + tail checksum, and write_at.
fn write_probe_roundtrip(block_size: u32) {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") {
        eprintln!("skipping: e2fsprogs tooling not available");
        return;
    }

    let probe_len = 10_000;
    let img = fresh_image(block_size, "written");
    let expected = payload(probe_len);

    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let ext4 = Ext4::open(dev);
        let mode = InodeFileType::S_IFREG.bits() | 0o644;
        let inode_ref = ext4
            .create(ROOT_INODE, "written.bin", mode)
            .expect("create /written.bin failed");
        let n = ext4
            .write_at(inode_ref.inode_num, 0, &expected)
            .expect("write_at /written.bin failed");
        assert_eq!(n, probe_len, "short write at block size {block_size}");
    }

    // The on-disk result must be a consistent filesystem.
    fsck_clean(&img);

    // Reopen and read it back through the crate.
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let ext4 = Ext4::open(dev);
    let inode = ext4
        .generic_open("/written.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("could not resolve /written.bin after write");
    let mut buf = vec![0u8; probe_len];
    let n = ext4
        .read_at(inode, 0, &mut buf)
        .expect("read_at /written.bin failed");
    assert_eq!(n, probe_len, "short readback at block size {block_size}");
    assert_eq!(buf, expected, "written content mismatch at block size {block_size}");
}

#[test]
fn write_probe_4k_baseline() {
    write_probe_roundtrip(4096);
}

#[test]
fn write_probe_2k() {
    write_probe_roundtrip(2048);
}

#[test]
fn write_probe_1k_target() {
    write_probe_roundtrip(1024);
}

// --- Hardening: exercise more write/metadata paths at non-4 KiB block sizes ---

fn open_fs(img: &Path) -> Ext4 {
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(img));
    Ext4::open(dev)
}

fn reg_mode() -> u16 {
    InodeFileType::S_IFREG.bits() | 0o644
}

fn tooling_ready() -> bool {
    !(tool_missing("mkfs.ext4") || tool_missing("e2fsck"))
}

/// Write a file in two calls (the second at EOF) and read the whole thing back.
fn append_roundtrip(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "append");
    let first = payload(5000);
    let second: Vec<u8> = payload(5000).iter().map(|b| b ^ 0xff).collect();

    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "a.bin", reg_mode()).expect("create");
        ext4.write_at(f.inode_num, 0, &first).expect("write 1");
        ext4.write_at(f.inode_num, first.len(), &second)
            .expect("append");
    }
    fsck_clean(&img);

    let mut expected = first.clone();
    expected.extend_from_slice(&second);
    let ext4 = open_fs(&img);
    let inode = ext4
        .generic_open("/a.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("resolve");
    let mut buf = vec![0u8; expected.len()];
    let n = ext4.read_at(inode, 0, &mut buf).expect("read");
    assert_eq!(n, expected.len(), "append short read @ {block_size}");
    assert_eq!(buf, expected, "append mismatch @ {block_size}");
}

/// Make a subdirectory, create a file in it, write, and read it back by path.
fn subdir_roundtrip(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "subdir");
    let data = payload(8000);

    {
        let ext4 = open_fs(&img);
        ext4.dir_mk("/sub").expect("mkdir /sub");
        let sub = ext4
            .generic_open("/sub", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("resolve /sub");
        let f = ext4.create(sub, "f.bin", reg_mode()).expect("create in subdir");
        ext4.write_at(f.inode_num, 0, &data).expect("write");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    let inode = ext4
        .generic_open("/sub/f.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("resolve /sub/f.bin");
    let mut buf = vec![0u8; data.len()];
    let n = ext4.read_at(inode, 0, &mut buf).expect("read");
    assert_eq!(n, data.len(), "subdir short read @ {block_size}");
    assert_eq!(buf, data, "subdir mismatch @ {block_size}");
}

/// Write a file, shrink it with truncate, verify the new size and contents.
fn truncate_roundtrip(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "truncate");
    let full = payload(10_000);
    let new_len = 3000usize;

    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "t.bin", reg_mode()).expect("create");
        ext4.write_at(f.inode_num, 0, &full).expect("write");
        let mut inode_ref = ext4.get_inode_ref(f.inode_num);
        ext4.truncate_inode(&mut inode_ref, new_len as u64)
            .expect("truncate");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    let inode = ext4
        .generic_open("/t.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("resolve");
    let mut buf = vec![0u8; full.len()];
    let n = ext4.read_at(inode, 0, &mut buf).expect("read");
    assert_eq!(n, new_len, "truncate size wrong @ {block_size}");
    assert_eq!(&buf[..new_len], &full[..new_len], "truncate content @ {block_size}");
}

/// Create a file, remove it, and confirm the filesystem stays consistent and
/// the path no longer resolves.
fn delete_clean(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "delete");
    let data = payload(6000);

    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "d.bin", reg_mode()).expect("create");
        ext4.write_at(f.inode_num, 0, &data).expect("write");
        ext4.file_remove("/d.bin").expect("remove");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    let r = ext4.generic_open("/d.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0);
    assert!(r.is_err(), "removed file still resolves @ {block_size}");
}

#[test]
fn append_1k() {
    append_roundtrip(1024);
}
#[test]
fn append_4k() {
    append_roundtrip(4096);
}
#[test]
fn subdir_1k() {
    subdir_roundtrip(1024);
}
#[test]
fn subdir_4k() {
    subdir_roundtrip(4096);
}
#[test]
fn truncate_1k() {
    truncate_roundtrip(1024);
}
#[test]
fn truncate_4k() {
    truncate_roundtrip(4096);
}
#[test]
fn delete_1k() {
    delete_clean(1024);
}
#[test]
fn delete_4k() {
    delete_clean(4096);
}

/// Make an empty directory, remove it, and confirm consistency at each step.
fn rmdir_roundtrip(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rmdir");

    {
        let ext4 = open_fs(&img);
        ext4.dir_mk("/rmd").expect("mkdir /rmd");
    }
    fsck_clean(&img); // a freshly-created empty directory must be valid

    {
        let ext4 = open_fs(&img);
        ext4.dir_remove(ROOT_INODE, "rmd").expect("rmdir /rmd");
    }
    fsck_clean(&img); // removing it must leave the filesystem consistent

    let ext4 = open_fs(&img);
    let r = ext4.generic_open("/rmd", &mut ROOT_INODE.clone(), false, 0, &mut 0);
    assert!(r.is_err(), "removed dir still resolves @ {block_size}");
}

/// Create a nested directory, then remove it leaf-first. Exercises rmdir of a
/// subdirectory whose parent is not the root (nested link-count accounting).
fn nested_rmdir(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "nested");

    {
        let ext4 = open_fs(&img);
        ext4.dir_mk("/a").expect("mkdir /a");
        ext4.dir_mk("/a/b").expect("mkdir /a/b");
    }
    fsck_clean(&img);

    {
        let ext4 = open_fs(&img);
        let a = ext4
            .generic_open("/a", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("resolve /a");
        ext4.dir_remove(a, "b").expect("rmdir /a/b");
    }
    fsck_clean(&img);

    {
        let ext4 = open_fs(&img);
        ext4.dir_remove(ROOT_INODE, "a").expect("rmdir /a");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    let r = ext4.generic_open("/a", &mut ROOT_INODE.clone(), false, 0, &mut 0);
    assert!(r.is_err(), "removed nested dir still resolves @ {block_size}");
}

/// rmdir on a non-empty directory must fail with ENOTEMPTY and leave the
/// filesystem (and the directory's contents) intact.
fn rmdir_nonempty(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rmdirne");

    {
        let ext4 = open_fs(&img);
        ext4.dir_mk("/d").expect("mkdir /d");
        let d = ext4
            .generic_open("/d", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("resolve /d");
        let f = ext4.create(d, "f.bin", reg_mode()).expect("create");
        ext4.write_at(f.inode_num, 0, &payload(2000)).expect("write");
    }
    fsck_clean(&img);

    {
        let ext4 = open_fs(&img);
        let r = ext4.dir_remove(ROOT_INODE, "d");
        assert!(r.is_err(), "rmdir of non-empty dir succeeded @ {block_size}");
        assert_eq!(
            r.unwrap_err().error(),
            Errno::ENOTEMPTY,
            "rmdir non-empty wrong errno @ {block_size}"
        );
    }
    fsck_clean(&img); // the failed rmdir must not have corrupted anything

    let ext4 = open_fs(&img);
    let r = ext4.generic_open("/d/f.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0);
    assert!(r.is_ok(), "/d/f.bin lost after failed rmdir @ {block_size}");
}

/// Write a file large enough to span a block-group boundary, then delete it.
/// At 1 KiB blocks a 16 MiB image is two groups (boundary ~8 MiB), so the
/// delete frees a range crossing the boundary, exercising balloc_free_blocks'
/// per-iteration group recomputation. e2fsck must stay clean.
fn multigroup_free(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "multigroup");
    let big = payload(9 * 1024 * 1024); // 9 MiB > the 8 MiB group boundary at 1 KiB

    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "big.bin", reg_mode()).expect("create");
        let n = ext4.write_at(f.inode_num, 0, &big).expect("write big.bin");
        assert_eq!(n, big.len(), "short write @ {block_size}");
    }
    fsck_clean(&img); // a large multi-group file is consistent

    {
        let ext4 = open_fs(&img);
        ext4.file_remove("/big.bin").expect("remove big.bin");
    }
    fsck_clean(&img); // the cross-group free leaves the filesystem consistent

    let ext4 = open_fs(&img);
    let r = ext4.generic_open("/big.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0);
    assert!(r.is_err(), "removed big.bin still resolves @ {block_size}");
}

#[test]
fn multigroup_free_1k() {
    multigroup_free(1024);
}

#[test]
fn nested_rmdir_1k() {
    nested_rmdir(1024);
}
#[test]
fn nested_rmdir_4k() {
    nested_rmdir(4096);
}
#[test]
fn rmdir_nonempty_1k() {
    rmdir_nonempty(1024);
}
#[test]
fn rmdir_nonempty_4k() {
    rmdir_nonempty(4096);
}

#[test]
fn rmdir_1k() {
    rmdir_roundtrip(1024);
}
#[test]
fn rmdir_4k() {
    rmdir_roundtrip(4096);
}

/// Create a symlink with the crate, require e2fsck to be clean, then reopen and
/// read the target back via readlink. A target under 60 bytes must be stored
/// inline in the inode (fast symlink); a longer target lives in a data block
/// (slow symlink). Either way readlink must return exactly the bytes given.
fn symlink_roundtrip(block_size: u32, tag: &str, target: &str) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, tag);

    {
        let mut ext4 = open_fs(&img);
        ext4.fuse_symlink(ROOT_INODE as u64, "link", target)
            .expect("symlink");
    }
    fsck_clean(&img); // a freshly-created symlink must be a valid filesystem

    let mut ext4 = open_fs(&img);
    let ino = ext4
        .generic_open("/link", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("resolve /link");
    let got = ext4.fuse_readlink(ino as u64).expect("readlink /link");
    assert_eq!(
        got,
        target.as_bytes(),
        "symlink target mismatch @ {block_size} (target len {})",
        target.len()
    );
}

// Short target: fits inline in i_block (fast symlink).
const FAST_TARGET: &str = "/etc/hostname";
// Long target: 80 bytes, must spill to a data block (slow symlink).
const SLOW_TARGET: &str =
    "/very/long/path/that/exceeds/sixty/bytes/and/must/live/in/a/data/block/xxxxxxx";

/// Create a regular file, hardlink it under a second name, and require e2fsck
/// to be clean: the on-disk link count must reflect both names. Both names must
/// resolve to the same inode, and the data must be readable via the new name.
fn hardlink_roundtrip(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "hardlink");
    let data = payload(4000);

    let orig_ino;
    {
        let mut ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "orig.bin", reg_mode()).expect("create");
        orig_ino = f.inode_num;
        ext4.write_at(f.inode_num, 0, &data).expect("write");
        ext4.fuse_link(f.inode_num as u64, ROOT_INODE as u64, "link.bin")
            .expect("link");
    }
    fsck_clean(&img); // link count must equal the number of names (2)

    let ext4 = open_fs(&img);
    let a = ext4
        .generic_open("/orig.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("resolve /orig.bin");
    let b = ext4
        .generic_open("/link.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("resolve /link.bin");
    assert_eq!(a, b, "hardlink resolves to a different inode @ {block_size}");
    assert_eq!(a, orig_ino, "hardlink points at the wrong inode @ {block_size}");

    let mut buf = vec![0u8; data.len()];
    let n = ext4.read_at(b, 0, &mut buf).expect("read via /link.bin");
    assert_eq!(n, data.len(), "hardlink short read @ {block_size}");
    assert_eq!(buf, data, "hardlink content mismatch @ {block_size}");
}

#[test]
fn hardlink_1k() {
    hardlink_roundtrip(1024);
}
#[test]
fn hardlink_4k() {
    hardlink_roundtrip(4096);
}

// --- rename ---

/// Resolve a path relative to root, returning the inode number or None.
fn resolve(ext4: &Ext4, path: &str) -> Option<u32> {
    ext4.generic_open(path, &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .ok()
}

/// Rename a file within the same directory to a name that doesn't yet exist.
/// The inode and its data must survive; the old name must disappear.
fn rename_file_samedir(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rn_file_same");
    let data = payload(4000);

    let ino;
    {
        let mut ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "a.bin", reg_mode()).expect("create");
        ino = f.inode_num;
        ext4.write_at(f.inode_num, 0, &data).expect("write");
        ext4.fuse_rename(ROOT_INODE as u64, "a.bin", ROOT_INODE as u64, "b.bin", 0)
            .expect("rename");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    assert!(resolve(&ext4, "/a.bin").is_none(), "old name still resolves @ {block_size}");
    let b = resolve(&ext4, "/b.bin").expect("new name missing");
    assert_eq!(b, ino, "renamed to a different inode @ {block_size}");

    let mut buf = vec![0u8; data.len()];
    let n = ext4.read_at(b, 0, &mut buf).expect("read renamed");
    assert_eq!(n, data.len(), "renamed short read @ {block_size}");
    assert_eq!(buf, data, "renamed content mismatch @ {block_size}");
}

#[test]
fn rename_file_samedir_1k() {
    rename_file_samedir(1024);
}
#[test]
fn rename_file_samedir_4k() {
    rename_file_samedir(4096);
}

/// Move a file from one directory to another (different parent). Its contents
/// must survive, the old path must disappear, and the new path must resolve.
fn rename_file_crossdir(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rn_file_cross");
    let data = payload(4000);

    {
        let mut ext4 = open_fs(&img);
        ext4.dir_mk("/src").expect("mkdir src");
        ext4.dir_mk("/dst").expect("mkdir dst");
        let src = resolve(&ext4, "/src").expect("resolve src");
        let f = ext4.create(src, "f.bin", reg_mode()).expect("create");
        ext4.write_at(f.inode_num, 0, &data).expect("write");
        let dst = resolve(&ext4, "/dst").expect("resolve dst");
        ext4.fuse_rename(src as u64, "f.bin", dst as u64, "f.bin", 0)
            .expect("rename");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    assert!(resolve(&ext4, "/src/f.bin").is_none(), "old path resolves @ {block_size}");
    let b = resolve(&ext4, "/dst/f.bin").expect("new path missing");
    let mut buf = vec![0u8; data.len()];
    ext4.read_at(b, 0, &mut buf).expect("read");
    assert_eq!(buf, data, "content mismatch @ {block_size}");
}

#[test]
fn rename_file_crossdir_1k() {
    rename_file_crossdir(1024);
}
#[test]
fn rename_file_crossdir_4k() {
    rename_file_crossdir(4096);
}

/// Move a non-empty directory to a different parent. The '..' entry must be
/// repointed and the parents' link counts adjusted, or e2fsck flags the image.
fn rename_dir_crossdir(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rn_dir_cross");
    let data = payload(3000);

    {
        let mut ext4 = open_fs(&img);
        ext4.dir_mk("/src").expect("mkdir src");
        ext4.dir_mk("/dst").expect("mkdir dst");
        ext4.dir_mk("/src/d").expect("mkdir src/d");
        let d = resolve(&ext4, "/src/d").expect("resolve d");
        let f = ext4.create(d, "inside.bin", reg_mode()).expect("create");
        ext4.write_at(f.inode_num, 0, &data).expect("write");

        let src = resolve(&ext4, "/src").expect("resolve src");
        let dst = resolve(&ext4, "/dst").expect("resolve dst");
        ext4.fuse_rename(src as u64, "d", dst as u64, "d", 0)
            .expect("rename dir");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    assert!(resolve(&ext4, "/src/d").is_none(), "old dir path resolves @ {block_size}");
    let moved = resolve(&ext4, "/dst/d").expect("moved dir missing");
    let inside = resolve(&ext4, "/dst/d/inside.bin").expect("moved file missing");
    let mut buf = vec![0u8; data.len()];
    ext4.read_at(inside, 0, &mut buf).expect("read inside");
    assert_eq!(buf, data, "moved content mismatch @ {block_size}");

    // '..' inside the moved dir must now point at /dst.
    let dst = resolve(&ext4, "/dst").expect("resolve dst");
    let dotdot = resolve(&ext4, "/dst/d/..").expect("resolve ..");
    assert_eq!(dotdot, dst, "'..' not repointed @ {block_size}");
    let _ = moved;
}

#[test]
fn rename_dir_crossdir_1k() {
    rename_dir_crossdir(1024);
}
#[test]
fn rename_dir_crossdir_4k() {
    rename_dir_crossdir(4096);
}

/// Rename a file onto an existing file: the destination is replaced (its inode
/// freed), and the new name carries the source's contents.
fn rename_replace_file(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rn_repl_file");
    let src_data = payload(4000);
    let dst_data = payload(2000);

    {
        let mut ext4 = open_fs(&img);
        let a = ext4.create(ROOT_INODE, "a.bin", reg_mode()).expect("create a");
        ext4.write_at(a.inode_num, 0, &src_data).expect("write a");
        let b = ext4.create(ROOT_INODE, "b.bin", reg_mode()).expect("create b");
        ext4.write_at(b.inode_num, 0, &dst_data).expect("write b");
        ext4.fuse_rename(ROOT_INODE as u64, "a.bin", ROOT_INODE as u64, "b.bin", 0)
            .expect("rename over");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    assert!(resolve(&ext4, "/a.bin").is_none(), "source survived @ {block_size}");
    let b = resolve(&ext4, "/b.bin").expect("dest missing");
    let mut buf = vec![0u8; src_data.len()];
    ext4.read_at(b, 0, &mut buf).expect("read");
    assert_eq!(buf, src_data, "dest not replaced with source @ {block_size}");
}

#[test]
fn rename_replace_file_1k() {
    rename_replace_file(1024);
}
#[test]
fn rename_replace_file_4k() {
    rename_replace_file(4096);
}

/// Rename a directory onto an existing empty directory: allowed, the empty
/// target is removed and replaced by the source directory.
fn rename_replace_empty_dir(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rn_repl_dir");
    let data = payload(3000);

    {
        let mut ext4 = open_fs(&img);
        ext4.dir_mk("/a").expect("mkdir a");
        let a = resolve(&ext4, "/a").expect("resolve a");
        let f = ext4.create(a, "inside.bin", reg_mode()).expect("create inside");
        ext4.write_at(f.inode_num, 0, &data).expect("write");
        ext4.dir_mk("/b").expect("mkdir b (empty target)");
        ext4.fuse_rename(ROOT_INODE as u64, "a", ROOT_INODE as u64, "b", 0)
            .expect("rename dir over empty dir");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    assert!(resolve(&ext4, "/a").is_none(), "source dir survived @ {block_size}");
    let inside = resolve(&ext4, "/b/inside.bin").expect("moved file missing");
    let mut buf = vec![0u8; data.len()];
    ext4.read_at(inside, 0, &mut buf).expect("read");
    assert_eq!(buf, data, "content mismatch @ {block_size}");
}

#[test]
fn rename_replace_empty_dir_1k() {
    rename_replace_empty_dir(1024);
}
#[test]
fn rename_replace_empty_dir_4k() {
    rename_replace_empty_dir(4096);
}

/// Error cases: the filesystem must stay clean and unchanged when a rename is
/// rejected.
fn rename_errors(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rn_errors");

    {
        let mut ext4 = open_fs(&img);
        // /file is a regular file; /dir is a non-empty directory.
        ext4.create(ROOT_INODE, "file", reg_mode()).expect("create file");
        ext4.dir_mk("/dir").expect("mkdir dir");
        let dir = resolve(&ext4, "/dir").expect("resolve dir");
        ext4.create(dir, "child", reg_mode()).expect("create child");
        ext4.dir_mk("/emptydir").expect("mkdir emptydir");

        // Source does not exist -> ENOENT.
        let e = ext4
            .fuse_rename(ROOT_INODE as u64, "nope", ROOT_INODE as u64, "x", 0)
            .unwrap_err();
        assert_eq!(e.error(), Errno::ENOENT, "missing source errno @ {block_size}");

        // File onto an existing directory -> EISDIR.
        let e = ext4
            .fuse_rename(ROOT_INODE as u64, "file", ROOT_INODE as u64, "dir", 0)
            .unwrap_err();
        assert_eq!(e.error(), Errno::EISDIR, "file-over-dir errno @ {block_size}");

        // Directory onto an existing file -> ENOTDIR.
        let e = ext4
            .fuse_rename(ROOT_INODE as u64, "dir", ROOT_INODE as u64, "file", 0)
            .unwrap_err();
        assert_eq!(e.error(), Errno::ENOTDIR, "dir-over-file errno @ {block_size}");

        // Directory onto a non-empty directory -> ENOTEMPTY.
        let e = ext4
            .fuse_rename(ROOT_INODE as u64, "emptydir", ROOT_INODE as u64, "dir", 0)
            .unwrap_err();
        assert_eq!(e.error(), Errno::ENOTEMPTY, "onto-nonempty errno @ {block_size}");
    }
    fsck_clean(&img); // every rejected rename must leave the image consistent

    // Nothing moved.
    let ext4 = open_fs(&img);
    assert!(resolve(&ext4, "/file").is_some(), "file vanished @ {block_size}");
    assert!(resolve(&ext4, "/dir/child").is_some(), "dir/child vanished @ {block_size}");
    assert!(resolve(&ext4, "/emptydir").is_some(), "emptydir vanished @ {block_size}");
}

#[test]
fn rename_errors_1k() {
    rename_errors(1024);
}
#[test]
fn rename_errors_4k() {
    rename_errors(4096);
}

const RENAME_NOREPLACE: u32 = 1;
const RENAME_EXCHANGE: u32 = 2;

/// RENAME_NOREPLACE must refuse to clobber an existing destination (EEXIST) but
/// still work when the destination is free.
fn rename_noreplace(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rn_noreplace");

    {
        let mut ext4 = open_fs(&img);
        ext4.create(ROOT_INODE, "a.bin", reg_mode()).expect("create a");
        ext4.create(ROOT_INODE, "b.bin", reg_mode()).expect("create b");

        // Destination exists -> EEXIST, nothing changes.
        let e = ext4
            .fuse_rename(ROOT_INODE as u64, "a.bin", ROOT_INODE as u64, "b.bin", RENAME_NOREPLACE)
            .unwrap_err();
        assert_eq!(e.error(), Errno::EEXIST, "noreplace errno @ {block_size}");

        // Destination free -> succeeds.
        ext4.fuse_rename(ROOT_INODE as u64, "a.bin", ROOT_INODE as u64, "c.bin", RENAME_NOREPLACE)
            .expect("noreplace to free name");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    assert!(resolve(&ext4, "/a.bin").is_none(), "a.bin survived @ {block_size}");
    assert!(resolve(&ext4, "/b.bin").is_some(), "b.bin vanished @ {block_size}");
    assert!(resolve(&ext4, "/c.bin").is_some(), "c.bin missing @ {block_size}");
}

#[test]
fn rename_noreplace_1k() {
    rename_noreplace(1024);
}
#[test]
fn rename_noreplace_4k() {
    rename_noreplace(4096);
}

/// RENAME_EXCHANGE atomically swaps two existing names; no inode is freed.
fn rename_exchange_files(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rn_exch_file");
    let a_data = payload(4000);
    let b_data = payload(2000);

    {
        let mut ext4 = open_fs(&img);
        let a = ext4.create(ROOT_INODE, "a.bin", reg_mode()).expect("create a");
        ext4.write_at(a.inode_num, 0, &a_data).expect("write a");
        let b = ext4.create(ROOT_INODE, "b.bin", reg_mode()).expect("create b");
        ext4.write_at(b.inode_num, 0, &b_data).expect("write b");

        // Missing partner -> ENOENT.
        let e = ext4
            .fuse_rename(ROOT_INODE as u64, "a.bin", ROOT_INODE as u64, "nope", RENAME_EXCHANGE)
            .unwrap_err();
        assert_eq!(e.error(), Errno::ENOENT, "exchange-missing errno @ {block_size}");

        ext4.fuse_rename(ROOT_INODE as u64, "a.bin", ROOT_INODE as u64, "b.bin", RENAME_EXCHANGE)
            .expect("exchange");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    // Names persist but their contents are swapped.
    let a = resolve(&ext4, "/a.bin").expect("a.bin missing");
    let b = resolve(&ext4, "/b.bin").expect("b.bin missing");
    let mut buf = vec![0u8; b_data.len()];
    ext4.read_at(a, 0, &mut buf).expect("read a");
    assert_eq!(buf, b_data, "a.bin not swapped @ {block_size}");
    let mut buf = vec![0u8; a_data.len()];
    ext4.read_at(b, 0, &mut buf).expect("read b");
    assert_eq!(buf, a_data, "b.bin not swapped @ {block_size}");
}

#[test]
fn rename_exchange_files_1k() {
    rename_exchange_files(1024);
}
#[test]
fn rename_exchange_files_4k() {
    rename_exchange_files(4096);
}

/// RENAME_EXCHANGE of two directories in different parents: each '..' must end
/// up pointing at its new parent and the link counts must stay consistent.
fn rename_exchange_dirs(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rn_exch_dir");
    let d1 = payload(3000);
    let d2 = payload(1500);

    {
        let mut ext4 = open_fs(&img);
        ext4.dir_mk("/p").expect("mkdir p");
        ext4.dir_mk("/q").expect("mkdir q");
        ext4.dir_mk("/p/d1").expect("mkdir p/d1");
        ext4.dir_mk("/q/d2").expect("mkdir q/d2");
        let f1 = ext4.create(resolve(&ext4, "/p/d1").unwrap(), "f1", reg_mode()).expect("f1");
        ext4.write_at(f1.inode_num, 0, &d1).expect("write f1");
        let f2 = ext4.create(resolve(&ext4, "/q/d2").unwrap(), "f2", reg_mode()).expect("f2");
        ext4.write_at(f2.inode_num, 0, &d2).expect("write f2");

        let p = resolve(&ext4, "/p").unwrap();
        let q = resolve(&ext4, "/q").unwrap();
        ext4.fuse_rename(p as u64, "d1", q as u64, "d2", RENAME_EXCHANGE)
            .expect("exchange dirs");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    // The directory trees swapped places: /p/d1 now holds f2, /q/d2 holds f1.
    let mut buf = vec![0u8; d2.len()];
    ext4.read_at(resolve(&ext4, "/p/d1/f2").expect("p/d1/f2 missing"), 0, &mut buf).expect("read");
    assert_eq!(buf, d2, "/p/d1 not swapped @ {block_size}");
    let mut buf = vec![0u8; d1.len()];
    ext4.read_at(resolve(&ext4, "/q/d2/f1").expect("q/d2/f1 missing"), 0, &mut buf).expect("read");
    assert_eq!(buf, d1, "/q/d2 not swapped @ {block_size}");

    // Each '..' points at its (unchanged) parent name.
    assert_eq!(resolve(&ext4, "/p/d1/..").unwrap(), resolve(&ext4, "/p").unwrap(), "p/d1/.. @ {block_size}");
    assert_eq!(resolve(&ext4, "/q/d2/..").unwrap(), resolve(&ext4, "/q").unwrap(), "q/d2/.. @ {block_size}");
}

#[test]
fn rename_exchange_dirs_1k() {
    rename_exchange_dirs(1024);
}
#[test]
fn rename_exchange_dirs_4k() {
    rename_exchange_dirs(4096);
}

/// Moving a directory into itself or one of its own descendants must fail with
/// EINVAL and leave the tree intact (otherwise it would create a detached
/// loop).
fn rename_into_own_subtree(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rn_subtree");

    {
        let mut ext4 = open_fs(&img);
        ext4.dir_mk("/a").expect("mkdir a");
        ext4.dir_mk("/a/b").expect("mkdir a/b");
        let a = resolve(&ext4, "/a").unwrap();
        let ab = resolve(&ext4, "/a/b").unwrap();

        // Into itself.
        let e = ext4
            .fuse_rename(ROOT_INODE as u64, "a", a as u64, "x", 0)
            .unwrap_err();
        assert_eq!(e.error(), Errno::EINVAL, "into-self errno @ {block_size}");

        // Into a descendant.
        let e = ext4
            .fuse_rename(ROOT_INODE as u64, "a", ab as u64, "x", 0)
            .unwrap_err();
        assert_eq!(e.error(), Errno::EINVAL, "into-descendant errno @ {block_size}");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    assert!(resolve(&ext4, "/a/b").is_some(), "/a/b vanished @ {block_size}");
}

#[test]
fn rename_into_own_subtree_1k() {
    rename_into_own_subtree(1024);
}
#[test]
fn rename_into_own_subtree_4k() {
    rename_into_own_subtree(4096);
}

// --- deep / multi-level extent trees ---

/// Write `n` single-block regions at non-contiguous logical offsets (gaps
/// between them), so each becomes its own non-mergeable extent. With enough
/// extents the inode's 4-entry root overflows and the tree must grow to a
/// multi-level (index -> leaf) shape, splitting nodes as leaves fill. e2fsck
/// must stay clean and every block must read back.
fn fragmented_file(block_size: u32, n: usize, min_depth: u16) {
    fragmented_ordered(block_size, min_depth, &format!("frag{}", n), (0..n).collect());
}

/// Like `fragmented_file`, but writes the `n` regions in the given `order`,
/// then verifies every block reads back regardless of insertion order. The
/// final logical layout is identical; only the order of `write_at` calls
/// differs, which exercises mid/front insertion into existing nodes.
fn fragmented_ordered(block_size: u32, min_depth: u16, tag: &str, order: Vec<usize>) {
    if !tooling_ready() {
        return;
    }
    let n = order.len();
    let bs = block_size as usize;
    let img = fresh_image(block_size, tag);
    let mk = |i: usize| -> Vec<u8> { payload(bs).iter().map(|b| b ^ (i as u8)).collect() };

    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "frag.bin", reg_mode()).expect("create");
        for &i in &order {
            // logical blocks 0, 2, 4, ... — a hole between each keeps the
            // extents from merging.
            ext4.write_at(f.inode_num, i * 2 * bs, &mk(i)).expect("write");
        }
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    let ino = resolve(&ext4, "/frag.bin").expect("resolve");

    // The tree must actually have grown to the expected shape, else the test
    // isn't exercising the multi-level paths it claims to.
    let depth = ext4.get_inode_ref(ino).inode.root_extent_header().depth;
    assert!(depth >= min_depth, "extent tree depth {depth} < {min_depth} ({tag})");

    for i in 0..n {
        let mut buf = vec![0u8; bs];
        ext4.read_at(ino, i * 2 * bs, &mut buf).expect("read");
        assert_eq!(buf, mk(i), "extent {i} mismatch ({tag})");
    }
}

#[test]
fn fragmented_shallow_1k() {
    // ~8 extents: overflows the 4-entry root, depth 1, no leaf split.
    fragmented_file(1024, 8, 1);
}
#[test]
fn fragmented_deep_1k() {
    // >84 extents: a 1 KiB leaf fills and must split.
    fragmented_file(1024, 200, 1);
}
#[test]
fn fragmented_deep_4k() {
    fragmented_file(4096, 200, 1);
}
#[test]
fn fragmented_depth2_1k() {
    // Enough extents to fill the 4-entry root index and grow to depth 2
    // (root -> index -> leaf).
    fragmented_file(1024, 600, 2);
}
#[test]
fn fragmented_huge_1k() {
    // Enough extents that an internal (depth-1) index node fills and must
    // itself split, not just the leaves.
    fragmented_file(1024, 8000, 2);
}
#[test]
fn fragmented_descending_1k() {
    // Highest logical block written first: every later write inserts before
    // existing extents (front of nodes), repointing parent index keys.
    fragmented_ordered(1024, 2, "frag_desc", (0..600).rev().collect());
}
#[test]
fn fragmented_shuffled_1k() {
    // Deterministic permutation (coprime stride) so writes land in arbitrary
    // positions within nodes, not just the front or back.
    let n = 600usize;
    let order: Vec<usize> = (0..n).map(|k| (k * 137) % n).collect();
    fragmented_ordered(1024, 2, "frag_shuf", order);
}

/// Overwrite an existing region in place: the second write must reuse the
/// already-allocated blocks (no new allocation, no leaks) and leave the new
/// contents. Guards the write path's allocate-only-holes decision.
fn overwrite_in_place(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "overwrite");
    let a = payload(8000);
    let b: Vec<u8> = payload(8000).iter().map(|x| x ^ 0xa5).collect();

    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "o.bin", reg_mode()).expect("create");
        ext4.write_at(f.inode_num, 0, &a).expect("write a");
        ext4.write_at(f.inode_num, 0, &b).expect("overwrite b");
    }
    fsck_clean(&img); // reused blocks, no leaked/duplicate blocks

    let ext4 = open_fs(&img);
    let ino = resolve(&ext4, "/o.bin").expect("resolve");
    let mut buf = vec![0u8; b.len()];
    ext4.read_at(ino, 0, &mut buf).expect("read");
    assert_eq!(buf, b, "overwrite content wrong @ {block_size}");
}

#[test]
fn overwrite_in_place_1k() {
    overwrite_in_place(1024);
}
#[test]
fn overwrite_in_place_4k() {
    overwrite_in_place(4096);
}

#[test]
fn symlink_fast_1k() {
    symlink_roundtrip(1024, "symlink_fast", FAST_TARGET);
}
#[test]
fn symlink_fast_4k() {
    symlink_roundtrip(4096, "symlink_fast", FAST_TARGET);
}
#[test]
fn symlink_slow_1k() {
    symlink_roundtrip(1024, "symlink_slow", SLOW_TARGET);
}
#[test]
fn symlink_slow_4k() {
    symlink_roundtrip(4096, "symlink_slow", SLOW_TARGET);
}


// --- extended attributes ---

/// debugfs writes attributes (a small one stored in the inode body, a large one
/// spilled to a block); the crate must read both back, list them, and report
/// ENODATA for a missing name.
fn xattr_read(block_size: u32) {
    if !tooling_ready() || tool_missing("debugfs") {
        return;
    }
    let img = fresh_image(block_size, "xattr_read");
    let big = vec![b'A'; 200];
    let bigfile = Path::new("target").join("harness").join("xbig.bin");
    fs::write(&bigfile, &big).unwrap();

    {
        let ext4 = open_fs(&img);
        ext4.create(ROOT_INODE, "x.bin", reg_mode()).expect("create");
    }
    debugfs(&img, "ea_set /x.bin user.foo barbar");
    debugfs(&img, "ea_set /x.bin trusted.t hello");
    debugfs(&img, &format!("ea_set -f {} /x.bin user.big", bigfile.display()));
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    let ino = resolve(&ext4, "/x.bin").expect("resolve");

    assert_eq!(ext4.xattr_get(ino, "user.foo").unwrap(), b"barbar", "user.foo @ {block_size}");
    assert_eq!(ext4.xattr_get(ino, "trusted.t").unwrap(), b"hello", "trusted.t @ {block_size}");
    assert_eq!(ext4.xattr_get(ino, "user.big").unwrap(), big, "user.big @ {block_size}");

    let mut names = ext4.xattr_list(ino).unwrap();
    names.sort();
    assert_eq!(names, vec!["trusted.t", "user.big", "user.foo"], "list @ {block_size}");

    assert_eq!(
        ext4.xattr_get(ino, "user.missing").unwrap_err().error(),
        Errno::ENODATA,
        "missing @ {block_size}"
    );
}

#[test]
fn xattr_read_1k() {
    xattr_read(1024);
}
#[test]
fn xattr_read_4k() {
    xattr_read(4096);
}

/// The crate writes attributes into the inode body; e2fsck must stay clean,
/// debugfs must read them back, and the crate must round-trip them. Also checks
/// removal and the XATTR_CREATE/REPLACE flag semantics.
fn xattr_write_ibody(block_size: u32) {
    if !tooling_ready() || tool_missing("debugfs") {
        return;
    }
    let img = fresh_image(block_size, "xattr_wib");

    let ino;
    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "x.bin", reg_mode()).expect("create");
        ino = f.inode_num;
        ext4.xattr_set(ino, "user.a", b"alpha", 0).expect("set a");
        ext4.xattr_set(ino, "user.b", b"beta", 0).expect("set b");
        ext4.xattr_set(ino, "security.s", b"sek", 0).expect("set s");
    }
    fsck_clean(&img);

    // External oracle: debugfs sees the attributes the crate wrote.
    let listed = debugfs(&img, "ea_list /x.bin");
    for n in ["user.a", "user.b", "security.s"] {
        assert!(listed.contains(n), "debugfs missing {n} @ {block_size}:\n{listed}");
    }
    assert!(
        debugfs(&img, "ea_get /x.bin user.a").contains("alpha"),
        "debugfs ea_get user.a @ {block_size}"
    );

    {
        let ext4 = open_fs(&img);
        assert_eq!(ext4.xattr_get(ino, "user.a").unwrap(), b"alpha");
        assert_eq!(ext4.xattr_get(ino, "user.b").unwrap(), b"beta");
        assert_eq!(ext4.xattr_get(ino, "security.s").unwrap(), b"sek");

        // Flag semantics.
        assert_eq!(
            ext4.xattr_set(ino, "user.a", b"x", 1).unwrap_err().error(),
            Errno::EEXIST,
            "CREATE on existing @ {block_size}"
        );
        assert_eq!(
            ext4.xattr_set(ino, "user.none", b"x", 2).unwrap_err().error(),
            Errno::ENODATA,
            "REPLACE on missing @ {block_size}"
        );

        // Remove one.
        ext4.xattr_remove(ino, "user.b").expect("remove b");
        assert_eq!(
            ext4.xattr_remove(ino, "user.b").unwrap_err().error(),
            Errno::ENODATA,
            "remove missing @ {block_size}"
        );
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    assert_eq!(ext4.xattr_get(ino, "user.b").unwrap_err().error(), Errno::ENODATA);
    assert_eq!(ext4.xattr_get(ino, "user.a").unwrap(), b"alpha", "a survived removal");
    let mut names = ext4.xattr_list(ino).unwrap();
    names.sort();
    assert_eq!(names, vec!["security.s", "user.a"], "final list @ {block_size}");
}

#[test]
fn xattr_write_ibody_1k() {
    xattr_write_ibody(1024);
}
#[test]
fn xattr_write_ibody_4k() {
    xattr_write_ibody(4096);
}

/// The crate stores attributes in an external block when they overflow the
/// inode body (a large value, and many small ones). e2fsck validates the block
/// (header, hashes, checksum); debugfs and the crate read them back. Removing
/// everything frees the block and clears i_file_acl.
fn xattr_block(block_size: u32) {
    if !tooling_ready() || tool_missing("debugfs") {
        return;
    }
    let img = fresh_image(block_size, "xattr_blk");
    let big = vec![b'Z'; 300];

    let ino;
    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "x.bin", reg_mode()).expect("create");
        ino = f.inode_num;
        ext4.xattr_set(ino, "user.big", &big, 0).expect("set big");
        // Several small attrs that, together, overflow the inode body.
        for i in 0..8 {
            ext4.xattr_set(ino, &format!("user.k{i}"), format!("val{i}").as_bytes(), 0)
                .expect("set small");
        }
        // The block must actually be in use.
        assert_ne!(
            ext4.get_inode_ref(ino).inode.file_acl,
            0,
            "expected an external xattr block @ {block_size}"
        );
    }
    fsck_clean(&img);

    let listed = debugfs(&img, "ea_list /x.bin");
    assert!(listed.contains("user.big"), "debugfs missing user.big @ {block_size}:\n{listed}");
    assert!(listed.contains("user.k7"), "debugfs missing user.k7 @ {block_size}:\n{listed}");

    {
        let ext4 = open_fs(&img);
        assert_eq!(ext4.xattr_get(ino, "user.big").unwrap(), big, "big value @ {block_size}");
        assert_eq!(ext4.xattr_get(ino, "user.k3").unwrap(), b"val3", "k3 @ {block_size}");
        assert_eq!(ext4.xattr_list(ino).unwrap().len(), 9, "count @ {block_size}");
    }

    // Remove everything; the block must be freed and i_file_acl cleared.
    {
        let ext4 = open_fs(&img);
        ext4.xattr_remove(ino, "user.big").expect("rm big");
        for i in 0..8 {
            ext4.xattr_remove(ino, &format!("user.k{i}")).expect("rm small");
        }
        assert_eq!(
            ext4.get_inode_ref(ino).inode.file_acl,
            0,
            "xattr block not freed @ {block_size}"
        );
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    assert!(ext4.xattr_list(ino).unwrap().is_empty(), "attrs remain @ {block_size}");
}

#[test]
fn xattr_block_1k() {
    xattr_block(1024);
}
#[test]
fn xattr_block_4k() {
    xattr_block(4096);
}

// --- POSIX ACLs ---

/// Build a `system.posix_acl_access` value: version 2 header + 8-byte entries.
fn acl_value(entries: &[(u16, u16, u32)]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&2u32.to_le_bytes());
    for &(tag, perm, id) in entries {
        v.extend_from_slice(&tag.to_le_bytes());
        v.extend_from_slice(&perm.to_le_bytes());
        v.extend_from_slice(&id.to_le_bytes());
    }
    v
}

/// Store an access ACL via the crate and check that the permission decision
/// follows POSIX ACL rules (owner, named user, owning group, other, all bounded
/// by the mask), with a clean fallback when no ACL is present.
fn acl_enforce(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "acl");
    const UNDEF: u32 = 0xFFFF_FFFF;
    // owner rw, user:1001 r, group r, mask rw, other none.
    let acl = acl_value(&[
        (0x01, 0o6, UNDEF),
        (0x02, 0o4, 1001),
        (0x04, 0o4, UNDEF),
        (0x10, 0o6, UNDEF),
        (0x20, 0o0, UNDEF),
    ]);

    let ino;
    {
        let ext4 = open_fs(&img);
        let f = ext4
            .create_with_attr(ROOT_INODE, "x.bin", InodeFileType::S_IFREG.bits() | 0o640, 1000, 2000)
            .expect("create");
        ino = f.inode_num;
        ext4.xattr_set(ino, "system.posix_acl_access", &acl, 0).expect("set acl");
        ext4.create(ROOT_INODE, "noacl.bin", reg_mode()).expect("create noacl");
    }
    fsck_clean(&img);

    // R=4, W=2, X=1.
    let ext4 = open_fs(&img);
    let chk = |u, g, w| ext4.acl_access_check(ino, u, g, w);

    assert_eq!(chk(1000, 0, 4), Some(true), "owner read @ {block_size}");
    assert_eq!(chk(1000, 0, 2), Some(true), "owner write @ {block_size}");
    assert_eq!(chk(1000, 0, 1), Some(false), "owner exec denied @ {block_size}");

    assert_eq!(chk(1001, 7, 4), Some(true), "user:1001 read @ {block_size}");
    assert_eq!(chk(1001, 7, 2), Some(false), "user:1001 write denied (mask) @ {block_size}");

    assert_eq!(chk(9999, 2000, 4), Some(true), "group read @ {block_size}");
    assert_eq!(chk(9999, 2000, 2), Some(false), "group write denied @ {block_size}");

    assert_eq!(chk(5000, 5000, 4), Some(false), "other denied @ {block_size}");

    // No ACL: fall back to mode bits.
    let noacl = resolve(&ext4, "/noacl.bin").expect("resolve noacl");
    assert_eq!(ext4.acl_access_check(noacl, 1, 1, 4), None, "no-acl falls back @ {block_size}");

    // End-to-end through fuse_access (other has no read access).
    let mut ext4 = open_fs(&img);
    assert!(!ext4.fuse_access(ino as u64, 5000, 5000, 4, 0), "fuse_access other read @ {block_size}");
    assert!(ext4.fuse_access(ino as u64, 1000, 0, 6, 0), "fuse_access owner rw @ {block_size}");
}

#[test]
fn acl_enforce_1k() {
    acl_enforce(1024);
}
#[test]
fn acl_enforce_4k() {
    acl_enforce(4096);
}

// --- ea_inode (large xattr values in dedicated inodes) ---

/// Build a fresh image with extra mkfs feature options enabled.
fn fresh_image_opts(block_size: u32, tag: &str, features: &str) -> PathBuf {
    let dir = Path::new("target").join("harness");
    fs::create_dir_all(&dir).unwrap();
    let img = dir.join(format!("{}_{}.img", tag, block_size));
    let _ = fs::remove_file(&img);
    fs::write(&img, vec![0u8; 16 * 1024 * 1024]).unwrap();
    let status = Command::new("mkfs.ext4")
        .args(["-q", "-b", &block_size.to_string(), "-O", features, "-F"])
        .arg(&img)
        .status()
        .expect("mkfs.ext4 failed to spawn");
    assert!(status.success(), "mkfs.ext4 -O {features} failed");
    img
}

/// A value too large for the xattr block is stored in a dedicated inode
/// (ea_inode feature). e2fsck must validate it; the crate must read it back;
/// removing it must free the value inode.
fn xattr_ea_inode(block_size: u32) {
    if !tooling_ready() || tool_missing("debugfs") {
        return;
    }
    let img = fresh_image_opts(block_size, "xattr_eai", "ea_inode");
    let big = payload(5000);

    let ino;
    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "x.bin", reg_mode()).expect("create");
        ino = f.inode_num;
        ext4.xattr_set(ino, "user.big", &big, 0).expect("set big");
        ext4.xattr_set(ino, "user.small", b"s", 0).expect("set small");
    }
    fsck_clean(&img);

    {
        let ext4 = open_fs(&img);
        assert_eq!(ext4.xattr_get(ino, "user.big").unwrap(), big, "big value @ {block_size}");
        assert_eq!(ext4.xattr_get(ino, "user.small").unwrap(), b"s", "small @ {block_size}");
    }

    // Remove the large attribute: its value inode must be freed.
    {
        let ext4 = open_fs(&img);
        ext4.xattr_remove(ino, "user.big").expect("rm big");
        ext4.xattr_remove(ino, "user.small").expect("rm small");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    assert!(ext4.xattr_list(ino).unwrap().is_empty(), "attrs remain @ {block_size}");
}

#[test]
fn xattr_ea_inode_1k() {
    xattr_ea_inode(1024);
}
#[test]
fn xattr_ea_inode_4k() {
    xattr_ea_inode(4096);
}

// --- mknod: special file types (char/block device, FIFO, socket) ---

/// `new_encode_dev`-style packing of a (major, minor) pair, matching what the
/// kernel hands a FUSE server as `rdev`. For small numbers this collapses to
/// `(major << 8) | minor`.
fn mkdev(major: u32, minor: u32) -> u32 {
    (minor & 0xff) | (major << 8) | ((minor & !0xff) << 12)
}

/// Create char/block/FIFO/socket nodes through the crate's mknod path. e2fsck
/// must stay clean (special files carry no extent tree), and debugfs must report
/// the right type for each — and the right device major/minor for the device
/// nodes (whose number is stored in i_block, not discarded).
fn mknod_special(block_size: u32) {
    if !tooling_ready() || tool_missing("debugfs") {
        return;
    }
    let img = fresh_image(block_size, "mknod");

    // /dev/null is char 1:3; a block device 8:0 (sda-like).
    let cdev = mkdev(1, 3);
    let bdev = mkdev(8, 0);

    {
        let ext4 = open_fs(&img);
        ext4.fuse_mknod_with_attr(
            ROOT_INODE as u64,
            "cdev",
            InodeFileType::S_IFCHR.bits() as u32 | 0o644,
            0,
            cdev,
            0,
            0,
        )
        .expect("mknod cdev");
        ext4.fuse_mknod_with_attr(
            ROOT_INODE as u64,
            "bdev",
            InodeFileType::S_IFBLK.bits() as u32 | 0o644,
            0,
            bdev,
            0,
            0,
        )
        .expect("mknod bdev");
        ext4.fuse_mknod_with_attr(
            ROOT_INODE as u64,
            "fifo",
            InodeFileType::S_IFIFO.bits() as u32 | 0o644,
            0,
            0,
            0,
            0,
        )
        .expect("mknod fifo");
        ext4.fuse_mknod_with_attr(
            ROOT_INODE as u64,
            "sock",
            InodeFileType::S_IFSOCK.bits() as u32 | 0o644,
            0,
            0,
            0,
            0,
        )
        .expect("mknod sock");
    }
    fsck_clean(&img);

    let cdev_stat = debugfs(&img, "stat /cdev");
    assert!(
        cdev_stat.contains("character special"),
        "cdev wrong type @ {block_size}:\n{cdev_stat}"
    );
    assert!(
        cdev_stat.contains("Device major/minor number: 01:03"),
        "cdev wrong device number @ {block_size}:\n{cdev_stat}"
    );

    let bdev_stat = debugfs(&img, "stat /bdev");
    assert!(
        bdev_stat.contains("block special"),
        "bdev wrong type @ {block_size}:\n{bdev_stat}"
    );
    assert!(
        bdev_stat.contains("Device major/minor number: 08:00"),
        "bdev wrong device number @ {block_size}:\n{bdev_stat}"
    );

    let fifo_stat = debugfs(&img, "stat /fifo");
    assert!(
        fifo_stat.contains("FIFO"),
        "fifo wrong type @ {block_size}:\n{fifo_stat}"
    );

    let sock_stat = debugfs(&img, "stat /sock");
    assert!(
        sock_stat.contains("socket"),
        "sock wrong type @ {block_size}:\n{sock_stat}"
    );
}

#[test]
fn mknod_special_1k() {
    mknod_special(1024);
}
#[test]
fn mknod_special_4k() {
    mknod_special(4096);
}

// --- readdir: stable byte-offset cookies / chunked resumption ---

/// Run a batch of debugfs commands from a request file (one per line).
fn debugfs_script(img: &Path, commands: &str) {
    let script = Path::new("target").join("harness").join("ddscript.txt");
    fs::write(&script, commands).unwrap();
    let out = Command::new("debugfs")
        .arg("-w")
        .arg("-f")
        .arg(&script)
        .arg(img)
        .output()
        .expect("debugfs -f failed to spawn");
    assert!(out.status.success(), "debugfs script failed");
}

/// Enumerate a directory in small chunks, resuming each call from the previous
/// batch's last cookie (the way a FUSE binding does once its reply buffer
/// fills). The reassembled listing must equal a single-shot read of the whole
/// directory: every entry exactly once, in order, with strictly increasing
/// cookies. This is what the old "cookie == array index" scheme could not
/// guarantee.
///
/// The directory is populated with debugfs (not the crate) so it spans several
/// directory blocks regardless of block size, isolating the readdir path under
/// test from the crate's own directory-growth code.
fn readdir_chunked(block_size: u32) {
    if !tooling_ready() || tool_missing("debugfs") {
        return;
    }
    let img = fresh_image(block_size, "readdir");

    // Enough entries (with longish names) to span several directory blocks at
    // both 1 KiB and 4 KiB.
    let n = 300usize;
    let mut script = String::new();
    for i in 0..n {
        script.push_str(&format!("mkdir /entry_{i:04}\n"));
    }
    debugfs_script(&img, &script);
    fsck_clean(&img); // the debugfs-built directory is consistent

    let ext4 = open_fs(&img);

    // Single-shot baseline: names from offset 0.
    let full = ext4.fuse_readdir(ROOT_INODE as u64, 0, 0).expect("readdir full");
    let full_names: Vec<String> = full.iter().map(|e| e.entry.get_name()).collect();
    assert!(
        full_names.len() >= n + 2,
        "expected at least {} entries (incl . and ..), got {} @ {block_size}",
        n + 2,
        full_names.len()
    );

    // Chunked: take 7 at a time, resume from the last entry's next_offset.
    let chunk = 7usize;
    let mut offset: i64 = 0;
    let mut got: Vec<String> = Vec::new();
    let mut last_cookie: u64 = 0;
    loop {
        let batch = ext4
            .fuse_readdir(ROOT_INODE as u64, 0, offset)
            .expect("readdir chunk");
        if batch.is_empty() {
            break;
        }
        let take = batch.len().min(chunk);
        for e in &batch[..take] {
            got.push(e.entry.get_name());
            assert!(
                e.next_offset > last_cookie,
                "cookies not strictly increasing ({} after {}) @ {block_size}",
                e.next_offset,
                last_cookie
            );
            last_cookie = e.next_offset;
        }
        // Resume after the last entry we consumed.
        offset = batch[take - 1].next_offset as i64;
    }

    assert_eq!(
        got, full_names,
        "chunked readdir != single-shot @ {block_size}"
    );
}

#[test]
fn readdir_chunked_1k() {
    readdir_chunked(1024);
}
#[test]
fn readdir_chunked_4k() {
    readdir_chunked(4096);
}

// --- multi-block directory growth ---

/// Create enough entries with the crate that the directory grows past its first
/// block, then require e2fsck to be clean. Each directory leaf block carries a
/// tail checksum seeded with the directory's inode number; an appended block
/// (whose first entry is a regular file, not ".") must still be seeded with the
/// directory inode, or e2fsck reports "directory ... fails checksum". Every
/// created name must also still resolve afterwards.
fn dir_multiblock(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "dirgrow");

    // 300 short-named entries span several blocks at 1 KiB and more than one at
    // 4 KiB (~255 entries per 4 KiB block).
    let n = 300usize;
    {
        let ext4 = open_fs(&img);
        for i in 0..n {
            ext4.create(ROOT_INODE, &format!("f{i:04}"), reg_mode())
                .expect("create");
        }
    }
    fsck_clean(&img); // appended directory blocks must checksum correctly

    // Confirm the directory actually grew past one block.
    let ext4 = open_fs(&img);
    let size = ext4.get_inode_ref(ROOT_INODE).inode.size();
    assert!(
        size > block_size as u64,
        "root directory did not grow past one block ({size} bytes) @ {block_size}"
    );

    // Every name still resolves.
    for i in 0..n {
        assert!(
            resolve(&ext4, &format!("/f{i:04}")).is_some(),
            "entry f{i:04} missing @ {block_size}"
        );
    }
}

#[test]
fn dir_multiblock_1k() {
    dir_multiblock(1024);
}
#[test]
fn dir_multiblock_4k() {
    dir_multiblock(4096);
}

// --- superblock free-count accounting ---

/// Many single allocations in one session (creating entries allocates an inode
/// each, and growing the directory allocates a block each) must leave the
/// superblock's free inode/block counters correct. e2fsck exits 0 on a
/// superblock-summary mismatch, so assert on its report text rather than its
/// exit status.
fn free_counts_consistent(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "freecount");

    let n = 250usize;
    {
        let ext4 = open_fs(&img);
        for i in 0..n {
            ext4.create(ROOT_INODE, &format!("g{i:04}"), reg_mode())
                .expect("create");
        }
    }

    let report = fsck_output(&img);
    assert!(
        !report.contains("Free inodes count wrong"),
        "superblock free inode count drifted @ {block_size}:\n{report}"
    );
    assert!(
        !report.contains("Free blocks count wrong"),
        "superblock free blocks count drifted @ {block_size}:\n{report}"
    );
}

#[test]
fn free_counts_consistent_1k() {
    free_counts_consistent(1024);
}
#[test]
fn free_counts_consistent_4k() {
    free_counts_consistent(4096);
}

// --- readdirplus ---

/// readdirplus returns the same entries and resume cookies as readdir, with
/// each entry's stat attributes attached. Verify the entry/cookie set matches
/// readdir exactly and that the per-entry attrs (inode number, kind, size) are
/// correct, including the implicit "." / ".." directories.
fn readdirplus_attrs(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "rdplus");

    // Files with distinct, known sizes.
    let files = [("a.bin", 1000usize), ("b.bin", 5000), ("c.bin", 250)];
    {
        let ext4 = open_fs(&img);
        for (name, len) in files {
            let f = ext4.create(ROOT_INODE, name, reg_mode()).expect("create");
            ext4.write_at(f.inode_num, 0, &payload(len)).expect("write");
        }
    }
    fsck_clean(&img); // readdirplus is read-only

    let ext4 = open_fs(&img);

    // Same entries and cookies as plain readdir.
    let plain = ext4.fuse_readdir(ROOT_INODE as u64, 0, 0).expect("readdir");
    let plus = ext4.fuse_readdirplus(ROOT_INODE as u64, 0, 0).expect("readdirplus");
    assert_eq!(plain.len(), plus.len(), "readdirplus count != readdir @ {block_size}");
    for (p, q) in plain.iter().zip(plus.iter()) {
        assert_eq!(q.entry.get_name(), p.entry.get_name(), "name mismatch @ {block_size}");
        assert_eq!(q.next_offset, p.next_offset, "cookie mismatch @ {block_size}");
        // Each entry's attr describes the inode the entry points at.
        assert_eq!(q.attr.ino, q.entry.inode as u64, "attr.ino mismatch @ {block_size}");
    }

    // Per-entry attrs by name.
    let by_name = |n: &str| plus.iter().find(|e| e.entry.get_name() == n).cloned();

    for (name, len) in files {
        let e = by_name(name).unwrap_or_else(|| panic!("{name} missing @ {block_size}"));
        assert_eq!(e.attr.kind, InodeFileType::S_IFREG, "{name} kind @ {block_size}");
        assert_eq!(e.attr.size, len as u64, "{name} size @ {block_size}");
    }

    let dot = by_name(".").expect("'.' missing");
    assert_eq!(dot.attr.kind, InodeFileType::S_IFDIR, "'.' kind @ {block_size}");
    assert_eq!(dot.attr.ino, ROOT_INODE as u64, "'.' ino @ {block_size}");
    let dotdot = by_name("..").expect("'..' missing");
    assert_eq!(dotdot.attr.kind, InodeFileType::S_IFDIR, "'..' kind @ {block_size}");
}

#[test]
fn readdirplus_attrs_1k() {
    readdirplus_attrs(1024);
}
#[test]
fn readdirplus_attrs_4k() {
    readdirplus_attrs(4096);
}

// --- POSIX default ACL inheritance + umask on create ---

/// A directory with a (non-trivial) default ACL passes it to new children: each
/// child gets a masq'd access ACL, its mode is clamped to the default, and a
/// child directory also inherits the default ACL itself. Children created
/// through the fuse entry points (which carry umask) drive this.
fn default_acl_inherit(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "defacl");
    const UNDEF: u32 = 0xFFFF_FFFF;
    // Default ACL: owner r-x, user:1001 rw-, group r--, mask rw-, other ---.
    // Non-trivial (named user + mask), so children store an access ACL.
    let default_acl = acl_value(&[
        (0x01, 0o5, UNDEF),
        (0x02, 0o6, 1001),
        (0x04, 0o4, UNDEF),
        (0x10, 0o6, UNDEF),
        (0x20, 0o0, UNDEF),
    ]);

    {
        let mut ext4 = open_fs(&img);
        ext4.dir_mk("/p").expect("mkdir p");
        let p = resolve(&ext4, "/p").expect("resolve p");
        ext4.xattr_set(p, "system.posix_acl_default", &default_acl, 0)
            .expect("set default acl");

        // Request mode 0o777; the default ACL must clamp it.
        ext4.fuse_create(p as u64, "file", InodeFileType::S_IFREG.bits() as u32 | 0o777, 0, 0)
            .expect("create file");
        ext4.fuse_mkdir(p as u64, "sub", InodeFileType::S_IFDIR.bits() as u32, 0)
            .expect("mkdir sub");
        ext4.fuse_mknod(p as u64, "fifo", InodeFileType::S_IFIFO.bits() as u32 | 0o777, 0, 0)
            .expect("mknod fifo");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);

    // Regular file: access ACL stored, mode clamped to 0o560 (owner from
    // USER_OBJ=5, group from MASK=6, other from OTHER=0), enforcement works.
    let file = resolve(&ext4, "/p/file").expect("resolve file");
    assert!(
        ext4.xattr_get(file, "system.posix_acl_access").is_ok(),
        "file missing inherited access ACL @ {block_size}"
    );
    assert_eq!(
        ext4.get_inode_ref(file).inode.mode() & 0o777,
        0o560,
        "file mode not clamped to default ACL @ {block_size}"
    );
    assert_eq!(
        ext4.acl_access_check(file, 1001, 0, 2),
        Some(true),
        "user:1001 write should be allowed @ {block_size}"
    );
    assert_eq!(
        ext4.acl_access_check(file, 1001, 0, 1),
        Some(false),
        "user:1001 exec should be denied @ {block_size}"
    );
    // A file is not a directory: it must NOT carry a default ACL.
    assert_eq!(
        ext4.xattr_get(file, "system.posix_acl_default").unwrap_err().error(),
        Errno::ENODATA,
        "file should not have a default ACL @ {block_size}"
    );

    // Subdirectory: inherits the default ACL verbatim, and has its own access ACL.
    let sub = resolve(&ext4, "/p/sub").expect("resolve sub");
    assert_eq!(
        ext4.xattr_get(sub, "system.posix_acl_default").expect("sub default acl"),
        default_acl,
        "subdir did not inherit the default ACL @ {block_size}"
    );
    assert!(
        ext4.xattr_get(sub, "system.posix_acl_access").is_ok(),
        "subdir missing inherited access ACL @ {block_size}"
    );

    // FIFO: gets an access ACL but no default ACL.
    let fifo = resolve(&ext4, "/p/fifo").expect("resolve fifo");
    assert!(
        ext4.xattr_get(fifo, "system.posix_acl_access").is_ok(),
        "fifo missing inherited access ACL @ {block_size}"
    );
    assert_eq!(
        ext4.xattr_get(fifo, "system.posix_acl_default").unwrap_err().error(),
        Errno::ENODATA,
        "fifo should not have a default ACL @ {block_size}"
    );
}

/// With no default ACL on the parent, a create through the fuse layer applies
/// umask to the requested mode (previously umask was ignored).
fn umask_on_create(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "umask");

    {
        let mut ext4 = open_fs(&img);
        ext4.fuse_create(
            ROOT_INODE as u64,
            "u",
            InodeFileType::S_IFREG.bits() as u32 | 0o666,
            0o022,
            0,
        )
        .expect("create u");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    let u = resolve(&ext4, "/u").expect("resolve u");
    assert_eq!(
        ext4.get_inode_ref(u).inode.mode() & 0o777,
        0o644,
        "umask 0o022 not applied to 0o666 @ {block_size}"
    );
}

#[test]
fn default_acl_inherit_1k() {
    default_acl_inherit(1024);
}
#[test]
fn default_acl_inherit_4k() {
    default_acl_inherit(4096);
}
#[test]
fn umask_on_create_1k() {
    umask_on_create(1024);
}
#[test]
fn umask_on_create_4k() {
    umask_on_create(4096);
}

// --- lseek: SEEK_DATA / SEEK_HOLE on sparse files ---

/// Build a sparse file (block 0 and block 3 written, blocks 1-2 holes, size 4
/// blocks) and check the data/hole boundary search. SEEK_DATA/SEEK_HOLE return
/// the offset unchanged when it already sits in the requested region, jump to
/// the next region boundary otherwise, treat EOF as an implicit hole, and
/// report ENXIO at/after EOF.
fn lseek_data_hole(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "lseek");
    let bs = block_size as i64;

    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "sparse.bin", reg_mode()).expect("create");
        ext4.write_at(f.inode_num, 0, &payload(block_size as usize)).expect("write blk0");
        ext4.write_at(f.inode_num, (3 * bs) as usize, &payload(block_size as usize))
            .expect("write blk3");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    let ino = resolve(&ext4, "/sparse.bin").expect("resolve") as u64;

    const SEEK_DATA: i32 = 3;
    const SEEK_HOLE: i32 = 4;
    let data = |o: i64| ext4.fuse_lseek(ino, 0, o, SEEK_DATA);
    let hole = |o: i64| ext4.fuse_lseek(ino, 0, o, SEEK_HOLE);

    assert_eq!(data(0).unwrap(), 0, "DATA(0) @ {block_size}");
    assert_eq!(hole(0).unwrap(), bs, "HOLE(0) @ {block_size}");
    assert_eq!(data(bs).unwrap(), 3 * bs, "DATA(bs) skips hole run @ {block_size}");
    assert_eq!(data(2 * bs).unwrap(), 3 * bs, "DATA(2bs) skips hole run @ {block_size}");
    assert_eq!(hole(3 * bs).unwrap(), 4 * bs, "HOLE(3bs) == EOF @ {block_size}");
    assert_eq!(data(100).unwrap(), 100, "DATA inside data returns offset @ {block_size}");
    assert_eq!(hole(bs + 10).unwrap(), bs + 10, "HOLE inside hole returns offset @ {block_size}");
    assert_eq!(
        data(4 * bs).unwrap_err().error(),
        Errno::ENXIO,
        "DATA at EOF @ {block_size}"
    );
    assert_eq!(
        hole(4 * bs).unwrap_err().error(),
        Errno::ENXIO,
        "HOLE at EOF @ {block_size}"
    );
    assert_eq!(
        ext4.fuse_lseek(ino, 0, 0, 0).unwrap_err().error(),
        Errno::EINVAL,
        "bad whence @ {block_size}"
    );
}

#[test]
fn lseek_data_hole_1k() {
    lseek_data_hole(1024);
}
#[test]
fn lseek_data_hole_4k() {
    lseek_data_hole(4096);
}

// --- fallocate: allocate / KEEP_SIZE / PUNCH_HOLE ---

const FALLOC_KEEP_SIZE: i32 = 0x01;
const FALLOC_PUNCH_HOLE: i32 = 0x02;

/// Preallocation: mode 0 maps the range and extends the size, and the range
/// reads back as zeros; KEEP_SIZE reserves blocks without changing the size;
/// invalid/unsupported requests are rejected.
fn fallocate_alloc(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "falloc_a");
    let bs = block_size as i64;
    const SEEK_DATA: i32 = 3;

    let ino;
    let ino_keep;
    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "a.bin", reg_mode()).expect("create a");
        ino = f.inode_num;
        ext4.fuse_fallocate(ino as u64, 0, 0, 4 * bs, 0).expect("fallocate");

        let k = ext4.create(ROOT_INODE, "k.bin", reg_mode()).expect("create k");
        ino_keep = k.inode_num;
        ext4.fuse_fallocate(ino_keep as u64, 0, 0, 2 * bs, FALLOC_KEEP_SIZE)
            .expect("fallocate keep");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);

    // mode 0: size extended, range reads as zeros, blocks mapped.
    assert_eq!(
        ext4.get_inode_ref(ino).inode.size(),
        (4 * bs) as u64,
        "size after allocate @ {block_size}"
    );
    let mut buf = vec![0xffu8; (4 * bs) as usize];
    let n = ext4.read_at(ino, 0, &mut buf).expect("read");
    assert_eq!(n, (4 * bs) as usize, "short read @ {block_size}");
    assert!(buf.iter().all(|&b| b == 0), "allocated range not zero @ {block_size}");
    assert_eq!(
        ext4.fuse_lseek(ino as u64, 0, 0, SEEK_DATA).unwrap(),
        0,
        "allocated blocks not mapped @ {block_size}"
    );

    // KEEP_SIZE: size unchanged but blocks reserved.
    assert_eq!(
        ext4.get_inode_ref(ino_keep).inode.size(),
        0,
        "KEEP_SIZE must not change size @ {block_size}"
    );
    assert!(
        ext4.get_inode_ref(ino_keep).inode.blocks_count() > 0,
        "KEEP_SIZE did not reserve blocks @ {block_size}"
    );

    // Errors.
    assert_eq!(
        ext4.fuse_fallocate(ino as u64, 0, 0, 0, 0).unwrap_err().error(),
        Errno::EINVAL,
        "zero length @ {block_size}"
    );
    assert_eq!(
        ext4.fuse_fallocate(ino as u64, 0, 0, bs, FALLOC_PUNCH_HOLE).unwrap_err().error(),
        Errno::EINVAL,
        "punch without keep_size @ {block_size}"
    );
    assert_eq!(
        ext4.fuse_fallocate(ino as u64, 0, 0, bs, 0x08).unwrap_err().error(),
        Errno::ENOTSUP,
        "collapse_range unsupported @ {block_size}"
    );
}

/// Punch hole: an aligned interior punch frees blocks (read as zeros, hole
/// observable via lseek) while surrounding data and the file size survive; a
/// sub-block punch zeroes just the requested bytes in a mapped block.
fn fallocate_punch(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "falloc_p");
    let bs = block_size as i64;
    const SEEK_DATA: i32 = 3;
    const SEEK_HOLE: i32 = 4;

    let ino;
    let ino_partial;
    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "p.bin", reg_mode()).expect("create p");
        ino = f.inode_num;
        ext4.write_at(ino, 0, &vec![0xABu8; (4 * bs) as usize]).expect("write p");
        // Punch the middle two blocks [bs, 3*bs).
        ext4.fuse_fallocate(ino as u64, 0, bs, 2 * bs, FALLOC_PUNCH_HOLE | FALLOC_KEEP_SIZE)
            .expect("punch");

        // Sub-block punch: zero bytes [10, 20) of a single data block.
        let pp = ext4.create(ROOT_INODE, "pp.bin", reg_mode()).expect("create pp");
        ino_partial = pp.inode_num;
        ext4.write_at(ino_partial, 0, &vec![0xABu8; bs as usize]).expect("write pp");
        ext4.fuse_fallocate(ino_partial as u64, 0, 10, 10, FALLOC_PUNCH_HOLE | FALLOC_KEEP_SIZE)
            .expect("partial punch");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);

    // Size unchanged, punched region zero, surrounding data intact.
    assert_eq!(
        ext4.get_inode_ref(ino).inode.size(),
        (4 * bs) as u64,
        "punch changed size @ {block_size}"
    );
    let mut mid = vec![0xffu8; (2 * bs) as usize];
    ext4.read_at(ino, bs as usize, &mut mid).expect("read mid");
    assert!(mid.iter().all(|&b| b == 0), "punched region not zero @ {block_size}");
    let mut blk0 = vec![0u8; bs as usize];
    ext4.read_at(ino, 0, &mut blk0).expect("read blk0");
    assert!(blk0.iter().all(|&b| b == 0xAB), "block 0 data lost @ {block_size}");
    let mut blk3 = vec![0u8; bs as usize];
    ext4.read_at(ino, (3 * bs) as usize, &mut blk3).expect("read blk3");
    assert!(blk3.iter().all(|&b| b == 0xAB), "block 3 data lost @ {block_size}");
    // Hole observable.
    assert_eq!(
        ext4.fuse_lseek(ino as u64, 0, bs, SEEK_HOLE).unwrap(),
        bs,
        "SEEK_HOLE not at punched block @ {block_size}"
    );
    assert_eq!(
        ext4.fuse_lseek(ino as u64, 0, bs, SEEK_DATA).unwrap(),
        3 * bs,
        "SEEK_DATA past hole wrong @ {block_size}"
    );

    // Sub-block punch zeroed only [10, 20).
    let mut b = vec![0u8; bs as usize];
    ext4.read_at(ino_partial, 0, &mut b).expect("read pp");
    for i in 0..bs as usize {
        let expect = if (10..20).contains(&i) { 0u8 } else { 0xABu8 };
        assert_eq!(b[i], expect, "partial punch byte {i} @ {block_size}");
    }
}

#[test]
fn fallocate_alloc_1k() {
    fallocate_alloc(1024);
}
#[test]
fn fallocate_alloc_4k() {
    fallocate_alloc(4096);
}
/// Writing into a preallocated (unwritten) region must convert the written
/// blocks to initialized so the data reads back, while the untouched part of
/// the range stays zero (still unwritten). Writing the middle of one unwritten
/// extent splits it into unwritten head / initialized middle / unwritten tail.
fn write_after_fallocate(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "falloc_w");
    let bs = block_size as usize;

    let ino;
    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "w.bin", reg_mode()).expect("create");
        ino = f.inode_num;
        // Preallocate 4 blocks (one unwritten extent), size = 4*bs.
        ext4.fuse_fallocate(ino as u64, 0, 0, (4 * bs) as i64, 0).expect("fallocate");
        // Write real data into the middle block (block 1).
        ext4.write_at(ino, bs, &vec![0xCDu8; bs]).expect("write mid");
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    // Block 1 reads the written data; blocks 0, 2, 3 still read as zeros.
    let mut b1 = vec![0u8; bs];
    ext4.read_at(ino, bs, &mut b1).expect("read b1");
    assert!(b1.iter().all(|&b| b == 0xCD), "written block not initialized @ {block_size}");
    for blk in [0usize, 2, 3] {
        let mut z = vec![0xffu8; bs];
        ext4.read_at(ino, blk * bs, &mut z).expect("read z");
        assert!(z.iter().all(|&b| b == 0), "block {blk} should still read zero @ {block_size}");
    }
    assert_eq!(
        ext4.get_inode_ref(ino).inode.size(),
        (4 * bs) as u64,
        "size changed @ {block_size}"
    );
}

#[test]
fn write_after_fallocate_1k() {
    write_after_fallocate(1024);
}
#[test]
fn write_after_fallocate_4k() {
    write_after_fallocate(4096);
}

#[test]
fn fallocate_punch_1k() {
    fallocate_punch(1024);
}
#[test]
fn fallocate_punch_4k() {
    fallocate_punch(4096);
}

// --- copy_file_range ---

/// Copy a byte range between two files: a full copy, an offset sub-range copy,
/// a past-EOF copy that returns only the available bytes, and rejection of a
/// nonzero flag.
fn copy_file_range_basic(block_size: u32) {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(block_size, "cfr");
    let len = 10_000usize;
    let src_data = payload(len);

    let src;
    let dst;
    let dst2;
    let dst3;
    {
        let ext4 = open_fs(&img);
        let s = ext4.create(ROOT_INODE, "src.bin", reg_mode()).expect("create src");
        src = s.inode_num;
        ext4.write_at(src, 0, &src_data).expect("write src");

        let d = ext4.create(ROOT_INODE, "dst.bin", reg_mode()).expect("create dst");
        dst = d.inode_num;
        let n = ext4
            .fuse_copy_file_range(src as u64, 0, 0, dst as u64, 0, 0, len as u64, 0)
            .expect("full copy");
        assert_eq!(n, len, "full copy count @ {block_size}");

        // Sub-range with differing offsets: src[2000,5000) -> dst2 offset 1000.
        let d2 = ext4.create(ROOT_INODE, "dst2.bin", reg_mode()).expect("create dst2");
        dst2 = d2.inode_num;
        let n = ext4
            .fuse_copy_file_range(src as u64, 0, 2000, dst2 as u64, 0, 1000, 3000, 0)
            .expect("sub copy");
        assert_eq!(n, 3000, "sub copy count @ {block_size}");

        // Past-EOF: ask for more than the source has from offset 7000.
        let d3 = ext4.create(ROOT_INODE, "dst3.bin", reg_mode()).expect("create dst3");
        dst3 = d3.inode_num;
        let n = ext4
            .fuse_copy_file_range(src as u64, 0, 7000, dst3 as u64, 0, 0, 999_999, 0)
            .expect("eof copy");
        assert_eq!(n, len - 7000, "past-EOF copy count @ {block_size}");

        // Nonzero flag rejected.
        assert_eq!(
            ext4.fuse_copy_file_range(src as u64, 0, 0, dst as u64, 0, 0, 1, 1)
                .unwrap_err()
                .error(),
            Errno::EINVAL,
            "nonzero flag @ {block_size}"
        );
    }
    fsck_clean(&img);

    let ext4 = open_fs(&img);
    let mut buf = vec![0u8; len];
    ext4.read_at(dst, 0, &mut buf).expect("read dst");
    assert_eq!(buf, src_data, "full copy mismatch @ {block_size}");

    let mut sub = vec![0u8; 3000];
    ext4.read_at(dst2, 1000, &mut sub).expect("read dst2");
    assert_eq!(sub, src_data[2000..5000], "sub copy mismatch @ {block_size}");

    let mut eofbuf = vec![0u8; len - 7000];
    ext4.read_at(dst3, 0, &mut eofbuf).expect("read dst3");
    assert_eq!(eofbuf, src_data[7000..], "past-EOF copy mismatch @ {block_size}");
}

#[test]
fn copy_file_range_basic_1k() {
    copy_file_range_basic(1024);
}
#[test]
fn copy_file_range_basic_4k() {
    copy_file_range_basic(4096);
}

// ===========================================================================
// FUSE stub implementations: lifecycle, fsync, bmap, ioctl, locks
// ===========================================================================

/// A block device wrapper that counts flush() calls, to prove fsync forces a
/// device flush rather than relying on write-through alone.
struct CountingDevice {
    inner: FileBlockDevice,
    flushes: Arc<core::sync::atomic::AtomicUsize>,
}

impl BlockDevice for CountingDevice {
    fn read_offset(&self, offset: usize, len: usize) -> Vec<u8> {
        self.inner.read_offset(offset, len)
    }
    fn write_offset(&self, offset: usize, data: &[u8]) {
        self.inner.write_offset(offset, data)
    }
    fn flush(&self) {
        self.flushes.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    }
}

#[test]
fn lifecycle_ops_are_ok_noops() {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(4096, "lifecycle");
    let mut ext4 = open_fs(&img);
    let f = ext4.create(ROOT_INODE, "f.bin", reg_mode()).expect("create");
    let ino = f.inode_num as u64;

    // None of these panic, and each reports success for this write-through fs.
    assert_eq!(ext4.fuse_flush(ino, 0, 0).expect("flush"), 0);
    assert_eq!(
        ext4.fuse_release(ino, 0, 0, None, false).expect("release"),
        0
    );
    assert_eq!(
        ext4.fuse_releasedir(ROOT_INODE as u64, 0, 0)
            .expect("releasedir"),
        0
    );
}

#[test]
fn fsync_flushes_the_device() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(4096, "fsync");
    let flushes = Arc::new(AtomicUsize::new(0));
    let dev: Arc<dyn BlockDevice> = Arc::new(CountingDevice {
        inner: FileBlockDevice::new(&img),
        flushes: flushes.clone(),
    });
    let mut ext4 = Ext4::open(dev);
    let f = ext4.create(ROOT_INODE, "f.bin", reg_mode()).expect("create");
    ext4.write_at(f.inode_num, 0, &payload(8192)).expect("write");

    ext4.fuse_fsync(f.inode_num as u64, 0, false).expect("fsync");
    assert!(flushes.load(Ordering::SeqCst) >= 1, "fsync did not flush");

    ext4.fuse_fsyncdir(ROOT_INODE as u64, 0, false)
        .expect("fsyncdir");
    assert!(
        flushes.load(Ordering::SeqCst) >= 2,
        "fsyncdir did not flush"
    );
}

#[test]
fn bmap_maps_logical_to_physical() {
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(4096, "bmap");
    let ext4 = open_fs(&img);
    let f = ext4.create(ROOT_INODE, "f.bin", reg_mode()).expect("create");
    let ino32 = f.inode_num;
    // Three full 4 KiB blocks, contiguous.
    ext4.write_at(ino32, 0, &payload(4096 * 3)).expect("write");
    let ino = ino32 as u64;

    let iref = ext4.get_inode_ref(ino32);
    // At blocksize == fs block size, bmap returns the physical fs block directly.
    for lblk in 0..3u64 {
        let expected = ext4.get_pblock_idx(&iref, lblk as u32).expect("pblock");
        assert_eq!(ext4.fuse_bmap(ino, 4096, lblk).expect("bmap"), expected);
    }

    // A logical block past EOF is a hole -> 0.
    assert_eq!(ext4.fuse_bmap(ino, 4096, 1000).expect("bmap hole"), 0);

    // Caller blocksize finer than the fs block: idx counts 2 KiB units, so
    // idx=2 addresses fs logical block 1, and the result is in 2 KiB units too.
    let phys_lb1 = ext4.get_pblock_idx(&iref, 1).expect("pblock");
    assert_eq!(ext4.fuse_bmap(ino, 2048, 2).expect("bmap scaled"), phys_lb1 * 2);
}

#[test]
fn ioctl_flags_and_version() {
    const FS_IOC_GETFLAGS: u32 = 0x8008_6601;
    const FS_IOC_SETFLAGS: u32 = 0x4008_6602;
    const FS_IOC_GETVERSION: u32 = 0x8008_7601;
    const FS_NOATIME_FL: u32 = 0x0000_0080;
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(4096, "ioctl");
    let le = |v: &[u8]| u32::from_le_bytes(v[..4].try_into().unwrap());

    {
        let ext4 = open_fs(&img);
        let f = ext4.create(ROOT_INODE, "f.bin", reg_mode()).expect("create");
        let ino = f.inode_num as u64;

        // NOATIME starts clear.
        let out = ext4
            .fuse_ioctl(ino, 0, 0, FS_IOC_GETFLAGS, &[], 4)
            .expect("getflags");
        assert_eq!(le(&out) & FS_NOATIME_FL, 0);

        // Set it, and read it back within the session.
        ext4.fuse_ioctl(ino, 0, 0, FS_IOC_SETFLAGS, &FS_NOATIME_FL.to_le_bytes(), 0)
            .expect("setflags");
        let out2 = ext4
            .fuse_ioctl(ino, 0, 0, FS_IOC_GETFLAGS, &[], 4)
            .expect("getflags2");
        assert_eq!(le(&out2) & FS_NOATIME_FL, FS_NOATIME_FL);

        // GETVERSION returns the inode generation.
        let ver = ext4
            .fuse_ioctl(ino, 0, 0, FS_IOC_GETVERSION, &[], 4)
            .expect("getversion");
        assert_eq!(le(&ver), ext4.get_inode_ref(f.inode_num).inode.generation());

        // Unknown command -> ENOTTY.
        let err = ext4.fuse_ioctl(ino, 0, 0, 0xDEAD_BEEF, &[], 0).unwrap_err();
        assert_eq!(err.error(), Errno::ENOTTY);
    }

    // The flag change persisted to a consistent on-disk inode.
    fsck_clean(&img);
    let ext4 = open_fs(&img);
    let ino = ext4
        .generic_open("/f.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("reopen");
    let out = ext4
        .fuse_ioctl(ino as u64, 0, 0, FS_IOC_GETFLAGS, &[], 4)
        .expect("getflags after reopen");
    assert_eq!(le(&out) & FS_NOATIME_FL, FS_NOATIME_FL);
}

#[test]
fn posix_locks_conflict_and_release() {
    use ext4_rs::{F_RDLCK, F_UNLCK, F_WRLCK};
    if !tooling_ready() {
        return;
    }
    const A: u64 = 0xAAAA;
    const B: u64 = 0xBBBB;
    const C: u64 = 0xCCCC;

    let img = fresh_image(4096, "locks");
    let mut ext4 = open_fs(&img);
    let f = ext4.create(ROOT_INODE, "f.bin", reg_mode()).expect("create");
    let ino = f.inode_num as u64;

    // No locks yet: a probe reports F_UNLCK.
    assert_eq!(ext4.fuse_getlk(ino, 0, B, 0, 100, F_WRLCK, 2).unwrap().typ, F_UNLCK);

    // A takes a write lock on [0, 100].
    ext4.fuse_setlk(ino, 0, A, 0, 100, F_WRLCK, 1, false).expect("A wrlock");

    // B's probe over [50, 150] reports A's conflicting write lock.
    let c = ext4.fuse_getlk(ino, 0, B, 50, 150, F_WRLCK, 2).unwrap();
    assert_eq!((c.typ, c.pid, c.start, c.end), (F_WRLCK, 1, 0, 100));

    // B trying to acquire the conflicting range fails with EAGAIN.
    let e = ext4.fuse_setlk(ino, 0, B, 50, 150, F_WRLCK, 2, false).unwrap_err();
    assert_eq!(e.error(), Errno::EAGAIN);

    // Read locks from different owners coexist.
    ext4.fuse_setlk(ino, 0, A, 200, 300, F_RDLCK, 1, false).expect("A rdlock");
    ext4.fuse_setlk(ino, 0, B, 250, 350, F_RDLCK, 2, false).expect("B rdlock");
    assert_eq!(ext4.fuse_getlk(ino, 0, C, 250, 260, F_RDLCK, 3).unwrap().typ, F_UNLCK);
    // ...but a write probe over the read-locked region conflicts.
    assert_eq!(ext4.fuse_getlk(ino, 0, C, 250, 260, F_WRLCK, 3).unwrap().typ, F_RDLCK);

    // A releases its write lock; B can now take [0, 100].
    ext4.fuse_setlk(ino, 0, A, 0, 100, F_UNLCK, 1, false).expect("A unlock");
    ext4.fuse_setlk(ino, 0, B, 0, 100, F_WRLCK, 2, false).expect("B wrlock");

    // flush by A drops A's remaining locks (its read lock on [200, 300]); only
    // B's read lock [250, 350] survives in that region.
    ext4.fuse_flush(ino, 0, A).expect("flush A");
    let after = ext4.fuse_getlk(ino, 0, C, 200, 300, F_WRLCK, 3).unwrap();
    assert_eq!((after.typ, after.owner), (F_RDLCK, B));
}

#[test]
fn poll_reports_regular_file_always_ready() {
    const POLLIN: u32 = 0x001;
    const POLLPRI: u32 = 0x002;
    const POLLOUT: u32 = 0x004;
    if !tooling_ready() {
        return;
    }
    let img = fresh_image(4096, "poll");
    let ext4 = open_fs(&img);
    let f = ext4.create(ROOT_INODE, "f.bin", reg_mode()).expect("create");
    let ino = f.inode_num as u64;

    // A regular file never blocks: the requested read/write readiness is ready.
    assert_eq!(
        ext4.fuse_poll(ino, 0, 0, POLLIN | POLLOUT, 0).expect("poll"),
        POLLIN | POLLOUT
    );
    // Events we don't signal (POLLPRI, out-of-band) are not reported ready.
    assert_eq!(
        ext4.fuse_poll(ino, 0, 0, POLLIN | POLLPRI, 0).expect("poll"),
        POLLIN
    );
}

// ===========================================================================
// HTree directory indexing
// ===========================================================================

/// Oracle: ext4's own directory hash, via `debugfs dx_hash` (default zero seed).
/// Returns (major, minor). Parses "Hash of <name> is 0x.. (minor 0x..)".
fn debugfs_dx_hash(algo: &str, name: &str) -> (u32, u32) {
    let out = Command::new("debugfs")
        .arg("-R")
        .arg(format!("dx_hash -h {algo} {name}"))
        .output()
        .expect("debugfs dx_hash spawn");
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().find(|l| l.contains("Hash of")).expect("hash line");
    let after = line.split(" is ").nth(1).expect("is");
    let hex = |t: &str| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).unwrap();
    let major = hex(after.split_whitespace().next().unwrap());
    let minor = hex(after.split("minor ").nth(1).unwrap().trim_end_matches(')'));
    (major, minor)
}

#[test]
fn dx_hash_matches_debugfs() {
    if tool_missing("debugfs") {
        eprintln!("skipping: debugfs not available");
        return;
    }
    let seed = [0u32; 4]; // zero seed => kernel default constants
    let names = [
        "a",
        "foo",
        "hello",
        "testfile",
        "a-longer-filename.txt",
        "0123456789abcdef0123456789abcdef-and-then-some-more",
    ];
    for (algo, ver) in [("legacy", 0u8), ("half_md4", 1u8), ("tea", 2u8)] {
        for name in names {
            let (emaj, emin) = debugfs_dx_hash(algo, name);
            let (maj, min) = ext4_rs::ext4_dir_hash(name.as_bytes(), ver, seed);
            assert_eq!(maj, emaj, "{algo} major hash for {name:?}");
            assert_eq!(min, emin, "{algo} minor hash for {name:?}");
        }
    }
}

/// Build an image whose `/big` directory is HTree-indexed: populate it with `n`
/// files via `mkfs.ext4 -d`, then `e2fsck -fyD` rebuilds it into a hash index.
/// `long_names` pads each name so leaves fill faster, forcing a multi-level
/// (indirect_levels ≥ 1) tree with fewer entries. Returns the image path and
/// the file names.
fn build_htree_fixture(
    block_size: u32,
    n: usize,
    tag: &str,
    long_names: bool,
) -> (PathBuf, Vec<String>) {
    let dir = Path::new("target").join("harness");
    fs::create_dir_all(&dir).unwrap();

    let src = dir.join(format!("htsrc_{tag}_{block_size}"));
    let big = src.join("big");
    let _ = fs::remove_dir_all(&src);
    fs::create_dir_all(&big).unwrap();

    let pad = if long_names { "_".repeat(180) } else { String::new() };
    let mut names = Vec::with_capacity(n);
    for i in 0..n {
        let name = format!("file_{:04}.dat{pad}", i);
        fs::write(big.join(&name), b"x").unwrap();
        names.push(name);
    }

    let img = dir.join(format!("htree_{tag}_{block_size}.img"));
    let _ = fs::remove_file(&img);
    fs::write(&img, vec![0u8; 48 * 1024 * 1024]).unwrap();

    let status = Command::new("mkfs.ext4")
        .args(["-q", "-b", &block_size.to_string(), "-F", "-d"])
        .arg(&src)
        .arg(&img)
        .status()
        .expect("mkfs.ext4 -d spawn");
    assert!(status.success(), "mkfs.ext4 -d failed");

    // -D rebuilds directories into htree indexes; it reports changes (nonzero).
    let _ = Command::new("e2fsck")
        .args(["-fyD"])
        .arg(&img)
        .output()
        .expect("e2fsck -fyD spawn");

    (img, names)
}

#[test]
fn htree_read_lookup_1k() {
    if !tooling_ready() || tool_missing("debugfs") {
        return;
    }
    let (img, names) = build_htree_fixture(1024, 600, "read", false);
    let ext4 = open_fs(&img);

    let big = ext4
        .generic_open("/big", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("open /big") as u32;
    assert!(
        ext4.get_inode_ref(big).inode.is_index(),
        "fixture /big is not htree-indexed"
    );

    // Every entry is found by descending the index to its hash-assigned leaf.
    for name in &names {
        let mut res = Ext4DirSearchResult::new(Ext4DirEntry::default());
        ext4.dx_find_entry(big, name, &mut res)
            .unwrap_or_else(|_| panic!("dx_find_entry missed {name}"));
        assert!(res.dentry.inode != 0, "zero inode for {name}");
    }

    // A name that doesn't exist resolves to ENOENT, not a wrong leaf.
    let mut res = Ext4DirSearchResult::new(Ext4DirEntry::default());
    let miss = ext4.dx_find_entry(big, "nope_not_here.dat", &mut res);
    assert_eq!(miss.unwrap_err().error(), Errno::ENOENT);

    fsck_clean(&img);
}

/// `indirect_levels` of `path`'s htree root, parsed from `debugfs htree`.
fn debugfs_htree_levels(img: &Path, path: &str) -> u8 {
    let out = Command::new("debugfs")
        .arg("-R")
        .arg(format!("htree {path}"))
        .arg(img)
        .output()
        .expect("debugfs htree spawn");
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s
        .lines()
        .find(|l| l.contains("Indirect levels"))
        .expect("indirect levels line");
    line.rsplit(':').next().unwrap().trim().parse().unwrap()
}

#[test]
fn htree_read_lookup_multilevel_1k() {
    if !tooling_ready() || tool_missing("debugfs") {
        return;
    }
    // Long names fill leaves fast, so 700 entries overflow the root into a
    // depth-1 tree — exercising the dx_node descent, not just the root.
    let (img, names) = build_htree_fixture(1024, 700, "readml", true);
    assert!(
        debugfs_htree_levels(&img, "/big") >= 1,
        "fixture is not multi-level; dx_node descent would be untested"
    );

    let ext4 = open_fs(&img);
    let big = ext4
        .generic_open("/big", &mut ROOT_INODE.clone(), false, 0, &mut 0)
        .expect("open /big") as u32;

    for name in &names {
        let mut res = Ext4DirSearchResult::new(Ext4DirEntry::default());
        ext4.dx_find_entry(big, name, &mut res)
            .unwrap_or_else(|_| panic!("dx_find_entry missed {name}"));
        assert!(res.dentry.inode != 0, "zero inode for {name}");
    }

    let mut res = Ext4DirSearchResult::new(Ext4DirEntry::default());
    let miss = ext4.dx_find_entry(big, "file_9999.datnope", &mut res);
    assert_eq!(miss.unwrap_err().error(), Errno::ENOENT);

    fsck_clean(&img);
}
