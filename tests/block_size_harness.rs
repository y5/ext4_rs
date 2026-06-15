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

use ext4_rs::{BlockDevice, Ext4};

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

/// Build a fresh image at `block_size`, with a single known file written via
/// debugfs at `/probe.bin`. Returns the image path and the expected contents.
fn make_image(block_size: u32, probe_len: usize) -> (PathBuf, Vec<u8>) {
    let dir = Path::new("target").join("harness");
    fs::create_dir_all(&dir).unwrap();

    let img = dir.join(format!("fixture_{}.img", block_size));
    let probe = dir.join(format!("probe_{}.bin", block_size));
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
fn read_probe_1k_target() {
    read_probe_matches(1024);
}
