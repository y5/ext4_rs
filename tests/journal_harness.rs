//! Journal (jbd2) load harness.
//!
//! Builds a real ext4 image with `mkfs.ext4` (which always lays down a journal
//! on inode 8), then drives `Journal::load` against it and checks the parsed
//! journal superblock.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use ext4_rs::{
    assemble_descriptor_block, finalize_commit_csum, finalize_revoke_csum, jbd2_csum_seed,
    jbd2_data_block_csum, parse_descriptor_block, patch_journal_sb_head, verify_commit_csum,
    BlockDevice, BlockTag, CommitBlock, CrashPoint, Ext4, InodeFileType, Journal, JournalDevice,
    RevokeBlock,
    TagFormat, JBD2_FEATURE_INCOMPAT_64BIT, JBD2_FEATURE_INCOMPAT_CSUM_V2,
    JBD2_FEATURE_INCOMPAT_CSUM_V3, JBD2_FLAG_LAST_TAG,
};

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

fn tool_missing(name: &str) -> bool {
    Command::new(name)
        .arg("-V")
        .output()
        .map(|_| false)
        .unwrap_or(true)
}

/// e2fsck must report the image fully consistent.
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

fn reg_mode() -> u16 {
    InodeFileType::S_IFREG.bits() | 0o644
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

#[test]
fn journal_log_block_read_write_roundtrip_4k() {
    if tool_missing("mkfs.ext4") {
        eprintln!("skip: mkfs.ext4 missing");
        return;
    }
    let img = fresh_image(4096, "jlogrw");
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let fs = Ext4::open(dev);
    let j = Journal::load(&fs).expect("load ok").expect("journal present");

    // Write a recognizable pattern into log block index 5, read it back.
    let bs = fs.block_size();
    let mut data = vec![0u8; bs];
    for (i, b) in data.iter_mut().enumerate() {
        *b = ((i * 7 + 3) % 251) as u8;
    }
    j.write_log_block(&fs, 5, &data).expect("write ok");
    let back = j.read_log_block(&fs, 5).expect("read ok");
    assert_eq!(back, data);

    // Wrong-size write is rejected.
    assert!(j.write_log_block(&fs, 5, &vec![0u8; bs - 1]).is_err());
}

/// One synthetic transaction to lay into the log: a set of (final-blocknr, body)
/// data blocks plus an optional list of revoked block numbers.
struct SynthTxn {
    sequence: u32,
    blocks: Vec<(u64, Vec<u8>)>,
    revokes: Vec<u64>,
}

/// Pad/truncate `data` to exactly `bs` bytes (zero-filled).
fn block_body(data: &[u8], bs: usize) -> Vec<u8> {
    let mut b = vec![0u8; bs];
    let n = data.len().min(bs);
    b[..n].copy_from_slice(&data[..n]);
    b
}

/// Write `txns` into the journal log starting at log block `j.sb.first`, chaining
/// per txn: `[descriptor][data blocks...]( [revoke] )[commit]` (the revoke block
/// is emitted only when the txn has revokes). Each block is written via
/// `j.write_log_block`. Afterwards the on-disk journal superblock is patched so
/// the journal looks DIRTY / pre-recovery: `s_start = j.sb.first` and
/// `s_sequence = txns[0].sequence`. Writes ONLY into the journal log (and the
/// journal sb); it does not touch any block's final on-disk location.
///
/// Returns the next free log-block index.
fn stage_dirty_journal(fs: &Ext4, j: &Journal, txns: &[SynthTxn]) -> u32 {
    let fmt = TagFormat::from_features(j.sb.feature_incompat);
    let has_64bit = j.sb.feature_incompat & JBD2_FEATURE_INCOMPAT_64BIT != 0;
    let has_csum = j.sb.feature_incompat
        & (JBD2_FEATURE_INCOMPAT_CSUM_V2 | JBD2_FEATURE_INCOMPAT_CSUM_V3)
        != 0;
    let seed = jbd2_csum_seed(&j.sb.uuid);
    let bs = fs.block_size();

    let mut cursor = j.sb.first;

    for txn in txns {
        // Build one tag per data block, with the per-data-block tag checksum.
        let mut tags = Vec::with_capacity(txn.blocks.len());
        let mut bodies = Vec::with_capacity(txn.blocks.len());
        for (blocknr, data) in &txn.blocks {
            let body = block_body(data, bs);
            let checksum = jbd2_data_block_csum(seed, txn.sequence, &body);
            tags.push(BlockTag {
                blocknr: *blocknr,
                flags: 0,
                checksum,
            });
            bodies.push(body);
        }

        // [descriptor]
        let desc =
            assemble_descriptor_block(seed, txn.sequence, bs, fmt, has_64bit, has_csum, &tags);
        j.write_log_block(fs, cursor, &desc).expect("write descriptor");
        cursor += 1;

        // [data blocks...]
        for body in &bodies {
            j.write_log_block(fs, cursor, body).expect("write data block");
            cursor += 1;
        }

        // ( [revoke] ) — only when this txn revokes something.
        if !txn.revokes.is_empty() {
            let mut rev = RevokeBlock {
                sequence: txn.sequence,
                blocks: txn.revokes.clone(),
            }
            .emit(bs, has_64bit);
            if has_csum {
                finalize_revoke_csum(&mut rev, seed);
            }
            j.write_log_block(fs, cursor, &rev).expect("write revoke");
            cursor += 1;
        }

        // [commit]
        let mut commit = CommitBlock {
            sequence: txn.sequence,
            commit_sec: 0,
            commit_nsec: 0,
        }
        .emit(bs);
        if has_csum {
            finalize_commit_csum(&mut commit, seed);
        }
        j.write_log_block(fs, cursor, &commit).expect("write commit");
        cursor += 1;
    }

    // Patch the on-disk journal superblock (log block 0) so it looks dirty:
    // s_sequence = txns[0].sequence and s_start = j.sb.first. The in-place patch
    // avoids depending on emit() fidelity for unrelated fields.
    let mut sb_block = j.read_log_block(fs, 0).expect("read journal sb");
    patch_journal_sb_head(&mut sb_block, txns[0].sequence, j.sb.first);
    j.write_log_block(fs, 0, &sb_block).expect("write journal sb");

    cursor
}

#[test]
fn recovery_replays_committed_and_honors_revoke_4k() {
    if tool_missing("mkfs.ext4") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jreplay");
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let fs = Ext4::open(dev);
    let j = Journal::load(&fs).expect("load").expect("journal");
    let base = j.sb.sequence;
    let bs = fs.block_size();

    // Final locations near the image end, safely outside the journal area.
    let p = 4000u64; // replayed
    let q = 4001u64; // journaled by A then revoked by B → must NOT be replayed
    let r = 4002u64; // replayed

    let data_x = vec![0xAB; bs];
    let data_q = vec![0xC5; bs]; // the "stale" content that must NOT land at q
    let data_r = vec![0x7E; bs];
    let txns = vec![
        SynthTxn { sequence: base,   blocks: vec![(p, data_x.clone()), (q, data_q.clone())], revokes: vec![] },
        SynthTxn { sequence: base+1, blocks: vec![(r, data_r.clone())], revokes: vec![q] },
    ];
    stage_dirty_journal(&fs, &j, &txns);

    j.recover(&fs).expect("recover ok");

    // p and r were replayed to their final locations.
    assert_eq!(fs.block_device.read_offset(p as usize * bs, bs), data_x);
    assert_eq!(fs.block_device.read_offset(r as usize * bs, bs), data_r);
    // q was revoked at a >= sequence, so the stale data_q must NOT be there.
    assert_ne!(fs.block_device.read_offset(q as usize * bs, bs), data_q);

    // Journal cleared: s_start (BE @28) == 0.
    let sb_block = j.read_log_block(&fs, 0).unwrap();
    assert_eq!(u32::from_be_bytes(sb_block[28..32].try_into().unwrap()), 0);

    // Idempotent: a second recover is a no-op and leaves the journal clear.
    j.recover(&fs).expect("second recover ok");
    let sb2 = j.read_log_block(&fs, 0).unwrap();
    assert_eq!(u32::from_be_bytes(sb2[28..32].try_into().unwrap()), 0);
}

#[test]
fn recovery_same_txn_revoke_skips_block_4k() {
    if tool_missing("mkfs.ext4") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jrevself");
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let fs = Ext4::open(dev);
    let j = Journal::load(&fs).expect("load").expect("journal");
    let base = j.sb.sequence;
    let bs = fs.block_size();

    let p = 4010u64;             // plain replayed block (control)
    let q = 4011u64;             // journaled AND revoked in the SAME txn → must be skipped
    let data_p = vec![0x11; bs];
    let data_q = vec![0x22; bs]; // stale body that must NOT land at q

    // Pre-seed q's final location with a sentinel; recovery must leave it intact.
    let sentinel = vec![0x99; bs];
    fs.block_device.write_offset(q as usize * bs, &sentinel);

    // One txn that logs both p and q, and revokes q within the same txn.
    let txns = vec![
        SynthTxn { sequence: base, blocks: vec![(p, data_p.clone()), (q, data_q.clone())], revokes: vec![q] },
    ];
    stage_dirty_journal(&fs, &j, &txns);

    j.recover(&fs).expect("recover");

    // Control: p replayed.
    assert_eq!(fs.block_device.read_offset(p as usize * bs, bs), data_p);
    // Boundary: q revoked at rseq == txn.sequence → skipped; sentinel survives,
    // stale data_q never written. (A buggy `>` comparison would overwrite the
    // sentinel with data_q and fail this.)
    assert_eq!(fs.block_device.read_offset(q as usize * bs, bs), sentinel);
}

#[test]
fn recovery_scan_finds_last_commit_4k() {
    if tool_missing("mkfs.ext4") {
        eprintln!("skip: mkfs.ext4 missing");
        return;
    }
    let img = fresh_image(4096, "jscan");
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let fs = Ext4::open(dev);
    let j = Journal::load(&fs).expect("load").expect("journal");
    let base = j.sb.sequence;

    // Four single-block transactions; the 4th's commit will be corrupted.
    let txns = vec![
        SynthTxn { sequence: base,     blocks: vec![(1001, b"aaaa".to_vec())], revokes: vec![] },
        SynthTxn { sequence: base + 1, blocks: vec![(1002, b"bbbb".to_vec())], revokes: vec![] },
        SynthTxn { sequence: base + 2, blocks: vec![(1003, b"cccc".to_vec())], revokes: vec![] },
        SynthTxn { sequence: base + 3, blocks: vec![(1004, b"dddd".to_vec())], revokes: vec![] },
    ];
    stage_dirty_journal(&fs, &j, &txns);

    // Each single-block txn occupies 3 log blocks: [descriptor][data][commit].
    // The 4th txn's commit is at log index first + 3*3 + 2 = first + 11. Corrupt it.
    let first = j.sb.first;
    let commit4 = first + 11;
    let mut blk = j.read_log_block(&fs, commit4).unwrap();
    blk[0..4].copy_from_slice(&[0, 0, 0, 0]); // smash the magic -> invalid commit
    j.write_log_block(&fs, commit4, &blk).unwrap();

    let scan = j.scan(&fs).expect("scan ok");
    assert_eq!(scan.txns.len(), 3, "only 3 fully-committed txns");
    assert_eq!(scan.last_sequence, base + 2);
    // The 3 committed txns map their logged blocks to the right final locations.
    assert_eq!(scan.txns[0].blocks[0].final_block, 1001);
    assert_eq!(scan.txns[2].blocks[0].final_block, 1003);
}

#[test]
fn recovery_revoke_table_built_4k() {
    if tool_missing("mkfs.ext4") {
        eprintln!("skip: mkfs.ext4 missing");
        return;
    }
    let img = fresh_image(4096, "jrevoke");
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let fs = Ext4::open(dev);
    let j = Journal::load(&fs).expect("load").expect("journal");
    let base = j.sb.sequence;
    let txns = vec![
        SynthTxn { sequence: base,     blocks: vec![(2001, b"x".to_vec())], revokes: vec![777] },
        SynthTxn { sequence: base + 1, blocks: vec![(2002, b"y".to_vec())], revokes: vec![777, 888] },
    ];
    stage_dirty_journal(&fs, &j, &txns);
    let scan = j.scan(&fs).expect("scan");
    let table = j.build_revoke_table(&fs, &scan).expect("revoke table");
    assert_eq!(table.get(&777).copied(), Some(base + 1)); // highest revoking seq wins
    assert_eq!(table.get(&888).copied(), Some(base + 1));
    assert_eq!(table.get(&999).copied(), None);
}

#[test]
fn recovery_revoke_table_empty_when_no_revokes_4k() {
    if tool_missing("mkfs.ext4") {
        eprintln!("skip: mkfs.ext4 missing");
        return;
    }
    let img = fresh_image(4096, "jrevokeempty");
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let fs = Ext4::open(dev);
    let j = Journal::load(&fs).expect("load").expect("journal");
    let base = j.sb.sequence;
    let txns = vec![
        SynthTxn { sequence: base, blocks: vec![(3001, b"z".to_vec())], revokes: vec![] },
    ];
    stage_dirty_journal(&fs, &j, &txns);
    let scan = j.scan(&fs).expect("scan");
    let table = j.build_revoke_table(&fs, &scan).expect("revoke table");
    assert!(table.is_empty());
}

#[test]
fn recovery_scan_clean_journal_is_empty_4k() {
    if tool_missing("mkfs.ext4") {
        eprintln!("skip: mkfs.ext4 missing");
        return;
    }
    let img = fresh_image(4096, "jscanclean");
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let fs = Ext4::open(dev);
    let j = Journal::load(&fs).expect("load").expect("journal");
    // Fresh mkfs journal has s_start == 0 => nothing to recover.
    let scan = j.scan(&fs).expect("scan ok");
    assert_eq!(scan.txns.len(), 0);
}

#[test]
fn synth_dirty_journal_layout_4k() {
    if tool_missing("mkfs.ext4") {
        eprintln!("skip: mkfs.ext4 missing");
        return;
    }
    let img = fresh_image(4096, "jsynth");
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let fs = Ext4::open(dev);
    let j = Journal::load(&fs).expect("load ok").expect("journal present");

    let seq = j.sb.sequence;
    let txn = SynthTxn {
        sequence: seq,
        blocks: vec![(1234, b"hello-block".to_vec())],
        revokes: vec![],
    };
    let next = stage_dirty_journal(&fs, &j, &[txn]);
    // descriptor + 1 data + commit = 3 blocks consumed from j.sb.first.
    assert_eq!(next, j.sb.first + 3);

    let fmt = TagFormat::from_features(j.sb.feature_incompat);
    let has_64bit = j.sb.feature_incompat & JBD2_FEATURE_INCOMPAT_64BIT != 0;
    let has_csum = j.sb.feature_incompat
        & (JBD2_FEATURE_INCOMPAT_CSUM_V2 | JBD2_FEATURE_INCOMPAT_CSUM_V3)
        != 0;
    let seed = jbd2_csum_seed(&j.sb.uuid);
    let bs = fs.block_size();

    // Re-read the descriptor (log block `first`) and parse it back.
    let desc = j.read_log_block(&fs, j.sb.first).expect("read descriptor");
    let tags = parse_descriptor_block(&desc, fmt, has_64bit).expect("parse descriptor");
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0].blocknr, 1234);
    assert_ne!(tags[0].flags & JBD2_FLAG_LAST_TAG, 0);
    // Descriptor tail csum is only written (and meaningful) when the journal
    // carries a CSUM feature. Stock mkfs.ext4 journals have NO journal-level
    // incompat features, so this is normally skipped; it exercises the tail
    // when run against a csum-enabled journal.
    if has_csum {
        let stored = u32::from_be_bytes([desc[bs - 4], desc[bs - 3], desc[bs - 2], desc[bs - 1]]);
        let mut zeroed = desc.clone();
        zeroed[bs - 4..bs].copy_from_slice(&[0u8; 4]);
        assert_eq!(stored, ext4_rs::jbd2_block_csum(seed, &zeroed));
    }

    // Re-read the commit block (first + 2): parse sequence (+ verify csum when
    // the journal carries a CSUM feature).
    let commit = j.read_log_block(&fs, j.sb.first + 2).expect("read commit");
    if has_csum {
        assert!(verify_commit_csum(&commit, seed));
    }
    let cb = CommitBlock::parse(&commit).expect("parse commit");
    assert_eq!(cb.sequence, seq);

    // Re-read the journal superblock (log block 0): s_start@28 and s_sequence@24.
    let sb_block = j.read_log_block(&fs, 0).expect("read journal sb");
    let s_sequence = u32::from_be_bytes([sb_block[24], sb_block[25], sb_block[26], sb_block[27]]);
    let s_start = u32::from_be_bytes([sb_block[28], sb_block[29], sb_block[30], sb_block[31]]);
    assert_eq!(s_start, j.sb.first);
    assert_eq!(s_sequence, seq);
}

#[test]
fn recovering_open_clean_image_is_fsck_clean_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") {
        eprintln!("skip");
        return;
    }
    let img = fresh_image(4096, "jopenclean");
    let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
    let _fs = Ext4::open_and_recover(dev).expect("open_and_recover");
    fsck_clean(&img); // clean journal → recovery is a no-op → image still consistent
}

#[test]
fn commit_writes_and_checkpoints_clean_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jcommit");
    let content = {
        // Build a JournalDevice-wrapped fs, capture a file-create+write in a txn, commit it.
        let file_dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let jdev = Arc::new(JournalDevice::new(file_dev.clone(), 4096));
        let fs = Ext4::open(jdev.clone());
        let journal = Journal::load(&fs).expect("load").expect("journal");
        let content = vec![0x5A_u8; 4096];

        jdev.begin(journal.sb.sequence);
        let f = fs.create(ROOT_INODE, "j.bin", reg_mode()).expect("create");
        fs.write_at(f.inode_num, 0, &content).expect("write");
        let txn = jdev.end().expect("txn");
        assert!(!txn.is_empty());
        journal.commit(&fs, &txn).expect("commit");
        content
    };

    // 1) Reopen WITHOUT the journal wrapper; the checkpointed file is on disk.
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open(dev);
        let ino = fs.generic_open("/j.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0).expect("open");
        let mut buf = vec![0u8; content.len()];
        let n = fs.read_at(ino, 0, &mut buf).expect("read");
        assert_eq!(n, content.len());
        assert_eq!(buf, content, "checkpointed file content present");
        // journal is clean: s_start == 0.
        let j = Journal::load(&fs).expect("load").expect("journal");
        assert_eq!(j.sb.start, 0, "journal clean after commit+checkpoint");
    }

    // 2) e2fsck clean — the journaled+checkpointed create is a consistent change.
    fsck_clean(&img);
}

#[test]
fn dirty_journal_recovered_change_visible_and_fsck_clean_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") {
        eprintln!("skip");
        return;
    }
    let img = fresh_image(4096, "jrecmount");
    let bs = {
        // 1) Create /f.bin with OLD content (one block) and find its data block P.
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open(dev);
        let bs = fs.block_size();
        let f = fs.create(ROOT_INODE, "f.bin", reg_mode()).expect("create");
        let old = vec![0x4F; bs]; // 'O'
        fs.write_at(f.inode_num, 0, &old).expect("write old");
        let ir = fs.get_inode_ref(f.inode_num);
        let p = fs.get_pblock_idx(&ir, 0).expect("data block"); // physical block of logical 0
        assert!(p != 0, "freshly-written file's data block was not allocated");

        // 2) Synthesize a dirty journal logging NEW content into P.
        let j = Journal::load(&fs).expect("load").expect("journal");
        let base = j.sb.sequence;
        let new = vec![0x4E; bs]; // 'N'
        let txns = vec![SynthTxn {
            sequence: base,
            blocks: vec![(p, new.clone())],
            revokes: vec![],
        }];
        stage_dirty_journal(&fs, &j, &txns);
        bs
    };

    // 3) Open WITH recovery — replays NEW into P, clears the journal.
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("open_and_recover");
        // 4) The crate now reads NEW content from /f.bin.
        let inode = fs
            .generic_open("/f.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("open f");
        let mut buf = vec![0u8; bs];
        let n = fs.read_at(inode, 0, &mut buf).expect("read");
        assert_eq!(n, bs);
        assert_eq!(buf, vec![0x4E; bs], "recovered NEW content visible via crate");
    }

    // 5) e2fsck clean (journal was cleared by recovery; data block overwrite is consistent).
    fsck_clean(&img);
}

/// The core crash-recovery round-trip: a txn durably committed to the journal but
/// NOT yet checkpointed must be fully replayed by recovery on the next mount,
/// yielding a consistent (e2fsck-clean) fs with the change visible. Parameterized
/// over block size.
fn crash_before_checkpoint_recovers(block_size: u32) {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") {
        eprintln!("skip");
        return;
    }
    let img = fresh_image(block_size, "jcrash");
    let content = vec![0x3C_u8; block_size as usize];

    // 1) Capture a file create+write in a txn, then COMMIT but CRASH before checkpoint.
    {
        let file_dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let jdev = Arc::new(JournalDevice::new(file_dev.clone(), block_size as usize));
        let fs = Ext4::open(jdev.clone());
        let journal = Journal::load(&fs).expect("load").expect("journal");
        jdev.begin(journal.sb.sequence);
        let f = fs.create(ROOT_INODE, "crash.bin", reg_mode()).expect("create");
        fs.write_at(f.inode_num, 0, &content).expect("write");
        let txn = jdev.end().expect("txn");
        journal
            .commit_with_crash(&fs, &txn, CrashPoint::AfterCommitBlock)
            .expect("commit-crash");
        // The inner image does NOT have the file yet (no checkpoint); the journal is dirty.
    }

    // 2) Reopen WITH recovery — replays the committed txn into the fs.
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("open_and_recover");
        let ino = fs
            .generic_open("/crash.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("open");
        let mut buf = vec![0u8; content.len()];
        let n = fs.read_at(ino, 0, &mut buf).expect("read");
        assert_eq!(n, content.len());
        assert_eq!(buf, content, "recovered file content after crash-before-checkpoint");
        let j = Journal::load(&fs).expect("load").expect("journal");
        assert_eq!(j.sb.start, 0, "journal cleared after recovery");
    }

    // 3) e2fsck clean: recovery produced a consistent fs.
    fsck_clean(&img);
}

/// A crash PART-WAY through checkpointing leaves some final-location blocks
/// written and others not, with the journal still dirty. Recovery must replay the
/// WHOLE transaction — rewriting the already-written blocks with identical bytes
/// (idempotent) and completing the rest — yielding a consistent fs. Parameterized
/// over block size. Recovery itself must also be idempotent: a second mount is a
/// harmless no-op.
fn crash_mid_checkpoint_is_idempotent(block_size: u32) {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") { eprintln!("skip"); return; }
    let img = fresh_image(block_size, "jmidck");
    let content = vec![0x6D_u8; block_size as usize];

    // Commit, but crash after checkpointing only 1 block (the rest unwritten).
    {
        let file_dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let jdev = Arc::new(JournalDevice::new(file_dev.clone(), block_size as usize));
        let fs = Ext4::open(jdev.clone());
        let journal = Journal::load(&fs).expect("load").expect("journal");
        jdev.begin(journal.sb.sequence);
        let f = fs.create(ROOT_INODE, "mid.bin", reg_mode()).expect("create");
        fs.write_at(f.inode_num, 0, &content).expect("write");
        let txn = jdev.end().expect("txn");
        // crash after only 1 of the txn's blocks reached its final location.
        journal.commit_with_crash(&fs, &txn, CrashPoint::MidCheckpoint(1)).expect("commit-mid");
    }

    // Recovery replays the whole txn (idempotently re-writing the 1, completing the rest).
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("open_and_recover");
        let ino = fs.generic_open("/mid.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0).expect("open");
        let mut buf = vec![0u8; content.len()];
        let n = fs.read_at(ino, 0, &mut buf).expect("read");
        assert_eq!(n, content.len());
        assert_eq!(buf, content, "mid-checkpoint crash recovered fully");
        let j = Journal::load(&fs).expect("load").expect("journal");
        assert_eq!(j.sb.start, 0, "journal cleared after recovery");
    }

    // Running recovery AGAIN must be a harmless no-op (idempotent recovery).
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("second open_and_recover");
        let j = Journal::load(&fs).expect("load").expect("journal");
        assert_eq!(j.sb.start, 0, "journal still clean after a second recovery");
    }

    fsck_clean(&img);
}

#[test] fn crash_mid_checkpoint_is_idempotent_1k() { crash_mid_checkpoint_is_idempotent(1024); }
#[test] fn crash_mid_checkpoint_is_idempotent_2k() { crash_mid_checkpoint_is_idempotent(2048); }
#[test] fn crash_mid_checkpoint_is_idempotent_4k() { crash_mid_checkpoint_is_idempotent(4096); }

#[test]
fn revoke_recorded_in_txn_is_emitted_and_read_back_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jrevemit");
    let bs = 4096usize;

    // A block we'll "free" (revoke) during the transaction. Pre-seed a sentinel
    // at its final location; the revoke must not cause its content to change.
    let revoked = 3500u64;
    let sentinel = vec![0xA7u8; bs];
    {
        let raw: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        raw.write_offset(revoked as usize * bs, &sentinel);
    }

    // Capture a real fs change AND record a revoke for `revoked`, then commit
    // durably but crash before checkpoint so the dirty journal still holds the
    // committed transaction (including its revoke block).
    let content = vec![0x51u8; bs];
    let revoke_seq = {
        let file_dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let jdev = Arc::new(JournalDevice::new(file_dev.clone(), bs));
        let fs = Ext4::open(jdev.clone());
        let journal = Journal::load(&fs).expect("load").expect("journal");
        let seq = journal.sb.sequence;
        jdev.begin(seq);
        let f = fs.create(ROOT_INODE, "rev.bin", reg_mode()).expect("create");
        fs.write_at(f.inode_num, 0, &content).expect("write");
        jdev.revoke(revoked); // fs "frees" the metadata block `revoked`
        let txn = jdev.end().expect("txn");
        assert!(txn.revokes.contains(&revoked));
        journal.commit_with_crash(&fs, &txn, CrashPoint::AfterCommitBlock).expect("commit");
        seq
    };

    // EMIT proven by reading the revoke back via the real recovery scan path:
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open(dev); // plain open (no recovery yet)
        let journal = Journal::load(&fs).expect("load").expect("journal");
        let scan = journal.scan(&fs).expect("scan");
        let table = journal.build_revoke_table(&fs, &scan).expect("revoke table");
        assert_eq!(table.get(&revoked).copied(), Some(revoke_seq),
            "commit emitted a revoke block that the scan reads back");
    }

    // HONOR + consistency: recover the dirty journal; the fs change lands, the
    // revoked block keeps its sentinel (it was not journaled, just revoked), and
    // the result is e2fsck-clean.
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("recover");
        let ino = fs.generic_open("/rev.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0).expect("open");
        let mut buf = vec![0u8; content.len()];
        fs.read_at(ino, 0, &mut buf).expect("read");
        assert_eq!(buf, content, "fs change recovered");
        let raw: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let back = raw.read_offset(revoked as usize * bs, bs);
        assert_eq!(back, sentinel, "revoked (non-journaled) block keeps its content");
    }
    fsck_clean(&img);
}

#[test]
fn journaled_write_is_consistent_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jintwrite");
    let content = vec![0x77u8; 4096];
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_journaled(dev).expect("open_journaled");
        let f = fs.create(ROOT_INODE, "w.bin", reg_mode()).expect("create");
        fs.write_at(f.inode_num, 0, &content).expect("write"); // auto-journaled
    }
    // Reopen plainly; the write was committed+checkpointed.
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open(dev);
        let ino = fs.generic_open("/w.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0).expect("open");
        let mut buf = vec![0u8; content.len()];
        fs.read_at(ino, 0, &mut buf).expect("read");
        assert_eq!(buf, content);
        assert_eq!(Journal::load(&fs).unwrap().unwrap().sb.start, 0, "journal clean");
    }
    fsck_clean(&img);
}

#[test]
fn journaled_write_crash_before_checkpoint_recovers_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jintcrash");
    let content = vec![0x33u8; 4096];
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_journaled(dev).expect("open_journaled");
        let f = fs.create(ROOT_INODE, "wc.bin", reg_mode()).expect("create");
        // Make the write's commit crash before checkpoint.
        fs.journal_device.as_ref().unwrap().set_crash(CrashPoint::AfterCommitBlock);
        fs.write_at(f.inode_num, 0, &content).expect("write"); // commits to journal, crashes before checkpoint
    }
    // Recover on next mount: the write must be present and consistent.
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("recover");
        let ino = fs.generic_open("/wc.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0).expect("open");
        let mut buf = vec![0u8; content.len()];
        fs.read_at(ino, 0, &mut buf).expect("read");
        assert_eq!(buf, content, "write recovered after crash-before-checkpoint");
    }
    fsck_clean(&img);
}

// ---------------------------------------------------------------------------
// Journaled creation ops (Task 6.2): create/mkdir/symlink/link wrapped in a
// transaction. A crash at the op's own commit (AfterCommitBlock) leaves the
// txn durable-but-uncheckpointed; recovery on the next mount must replay it,
// yielding a consistent (e2fsck-clean) fs with the created object present.
// ---------------------------------------------------------------------------

#[test]
fn journaled_create_crash_recovers_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jcreate");
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_journaled(dev).expect("open_journaled");
        fs.journal_device.as_ref().unwrap().set_crash(CrashPoint::AfterCommitBlock);
        // `create` is itself journaled now; this commits then crashes pre-checkpoint.
        fs.create(ROOT_INODE, "f.bin", reg_mode()).expect("create");
    }
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("recover");
        let ino = fs
            .generic_open("/f.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("file present after recovery");
        assert!(ino >= 2);
    }
    fsck_clean(&img);
}

#[test]
fn journaled_mkdir_crash_recovers_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jmkdir");
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let mut fs = Ext4::open_journaled(dev).expect("open_journaled");
        fs.journal_device.as_ref().unwrap().set_crash(CrashPoint::AfterCommitBlock);
        fs.fuse_mkdir(
            ROOT_INODE as u64,
            "d",
            InodeFileType::S_IFDIR.bits() as u32,
            0,
        )
        .expect("mkdir");
    }
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("recover");
        let ino = fs
            .generic_open("/d", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("dir present");
        assert!(ino >= 2);
    }
    fsck_clean(&img);
}

#[test]
fn journaled_symlink_crash_recovers_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jsymlink");
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let mut fs = Ext4::open_journaled(dev).expect("open_journaled");
        fs.journal_device.as_ref().unwrap().set_crash(CrashPoint::AfterCommitBlock);
        fs.fuse_symlink(ROOT_INODE as u64, "lnk", "/some/target").expect("symlink");
    }
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("recover");
        let ino = fs
            .generic_open("/lnk", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("symlink present");
        assert!(ino >= 2);
    }
    fsck_clean(&img);
}

#[test]
fn journaled_link_crash_recovers_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jhardlink");
    let target_ino;
    {
        // Create the link target first (its own committed+checkpointed txn),
        // then hard-link to it with a crash at the link's commit.
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let mut fs = Ext4::open_journaled(dev).expect("open_journaled");
        let f = fs.create(ROOT_INODE, "tgt.bin", reg_mode()).expect("create target");
        target_ino = f.inode_num;
        fs.journal_device.as_ref().unwrap().set_crash(CrashPoint::AfterCommitBlock);
        fs.fuse_link(target_ino as u64, ROOT_INODE as u64, "hard").expect("link");
    }
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("recover");
        // both the original name and the new hard link resolve.
        let a = fs
            .generic_open("/tgt.bin", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("target present");
        let b = fs
            .generic_open("/hard", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("hard link present");
        assert_eq!(a, b, "hard link points at the same inode");
        assert_eq!(a, target_ino);
    }
    fsck_clean(&img);
}

/// A creation op that FAILS (mkdir of an existing name → EEXIST) on a journaled
/// fs must leave the fs consistent: the failing op's transaction is discarded
/// (atomic abort via `journal_end(false)`), so no stray object is created and
/// the result is e2fsck-clean.
#[test]
fn journaled_failed_op_aborts_cleanly_4k() {
    if tool_missing("mkfs.ext4") || tool_missing("e2fsck") { eprintln!("skip"); return; }
    let img = fresh_image(4096, "jabort");
    {
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let mut fs = Ext4::open_journaled(dev).expect("open_journaled");
        let mode = InodeFileType::S_IFDIR.bits() as u32;
        // First mkdir succeeds (committed + checkpointed).
        fs.fuse_mkdir(ROOT_INODE as u64, "dup", mode, 0).expect("first mkdir");
        // Second mkdir of the same name must fail with EEXIST → its txn aborts.
        let err = fs.fuse_mkdir(ROOT_INODE as u64, "dup", mode, 0);
        assert!(err.is_err(), "duplicate mkdir must fail");
    }
    {
        // Recover (clean journal) and confirm exactly one "dup" exists and the
        // fs is consistent.
        let dev: Arc<dyn BlockDevice> = Arc::new(FileBlockDevice::new(&img));
        let fs = Ext4::open_and_recover(dev).expect("recover");
        let ino = fs
            .generic_open("/dup", &mut ROOT_INODE.clone(), false, 0, &mut 0)
            .expect("the one good dir is present");
        assert!(ino >= 2);
    }
    fsck_clean(&img);
}

#[test]
fn crash_before_checkpoint_recovers_1k() {
    crash_before_checkpoint_recovers(1024);
}

#[test]
fn crash_before_checkpoint_recovers_2k() {
    crash_before_checkpoint_recovers(2048);
}

#[test]
fn crash_before_checkpoint_recovers_4k() {
    crash_before_checkpoint_recovers(4096);
}
