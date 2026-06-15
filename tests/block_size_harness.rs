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

use ext4_rs::{BlockDevice, Errno, Ext4, InodeFileType};

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
