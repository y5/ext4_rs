//! Journal (jbd2) load harness.
//!
//! Builds a real ext4 image with `mkfs.ext4` (which always lays down a journal
//! on inode 8), then drives `Journal::load` against it and checks the parsed
//! journal superblock.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use ext4_rs::{BlockDevice, Ext4, Journal};

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

#[test]
fn journal_loads_clean_superblock_4k() {
    if tool_missing("mkfs.ext4") {
        eprintln!("skip: mkfs.ext4 missing");
        return;
    }
    let img = fresh_image(4096, "jload");
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let fs = Ext4::open(dev);
    let j = Journal::load(&fs).expect("load ok").expect("journal present");
    assert_eq!(j.sb.blocksize as usize, 4096);
    assert!(j.sb.maxlen > 0);
    assert_eq!(j.sb.start, 0); // freshly mkfs'd journal is clean
}
