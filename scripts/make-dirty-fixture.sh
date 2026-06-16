#!/usr/bin/env bash
set -euo pipefail

# make-dirty-fixture.sh — capture an AUTHENTIC kernel-produced dirty jbd2 journal
# as a fixture for ext4_rs Tier-3 recovery verification.
#
# WHY THIS EXISTS
# ---------------
# Our recovery (Ext4::open_and_recover) has been tested against journals our OWN
# crate emits (crash-before-checkpoint, at V1 and forced-CSUM_V3). The remaining
# gap is journals written by a REAL Linux kernel: it lays down CSUM_V3
# multi-transaction journals with real escape sequences and tag layouts we don't
# generate ourselves. Capturing one of those means loop-mounting an image,
# writing into it, and snapshotting the raw device while the journal still holds
# committed-but-un-checkpointed transactions. That needs root, so this script is
# meant to be run by a privileged human / CI runner — NOT in the unprivileged
# dev sandbox.
#
# HONEST LIMITATION (READ THIS)
# -----------------------------
# A fully portable, root-free, *guaranteed*-authentic crash capture is not
# achievable from a shell script. A genuine power-cut leaves the journal dirty
# because the kernel committed transactions to the log but had not yet
# checkpointed them to their final locations. We approximate that here by:
#   1. mkfs + loop-mount an image,
#   2. write files/dirs (so the kernel logs real transactions),
#   3. snapshot the RAW image bytes while it is still mounted (so we capture the
#      log mid-flight, before checkpoint/unmount flushes everything),
#   4. lazy-unmount (umount -l) the original.
# The snapshot taken at step 3 is an inconsistent-but-journal-dirty image: its
# superblock should show needs_recovery and inode 8 should hold a non-empty log.
#
# This is best-effort. The result is ONLY trustworthy as a fixture if you VERIFY
# it (see VERIFY below) shows a dirty journal. If the kernel happened to
# checkpoint everything before the snapshot, the journal will be clean and the
# image is useless as a dirty fixture — re-run, add more/larger writes, or
# shorten the window before the snapshot.
#
# An even more authentic alternative, if your environment supports it: use a
# device-mapper "snapshot" / "dm-flakey" target or drop the loop device WITHOUT
# unmounting (echo 1 > /proc/sys/vm/drop_caches then detach), or run the writes
# inside a VM you hard-kill. Those are out of scope for a portable script but are
# the gold standard; this script documents and performs the pragmatic path.
#
# USAGE
# -----
#   sudo scripts/make-dirty-fixture.sh <output.img> [block_size]
#
#   <output.img>  path to write the captured fixture image to
#   [block_size]  ext4 block size in bytes (default 4096)
#
# After a successful capture, place the image at:
#   tests/fixtures/dirty/<name>.img
# and run the (otherwise ignored) replay test with:
#   EXT4_KERNEL_FIXTURES=1 cargo test --test journal_harness kernel_dirty -- --ignored

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <output.img> [block_size]" >&2
    exit 2
fi

OUT="$1"
BS="${2:-4096}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "error: must run as root (loop-mount + raw snapshot require privileges)" >&2
    echo "       re-run with: sudo $0 $*" >&2
    exit 1
fi

for tool in mkfs.ext4 dumpe2fs debugfs losetup mount umount; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool '$tool' not found in PATH" >&2
        exit 1
    fi
done

# Working image we mount + dirty. We snapshot a COPY of this into $OUT.
WORK="$(mktemp --suffix=.img)"
MNT="$(mktemp -d)"
LOOP=""

cleanup() {
    # Best-effort teardown. We lazy-unmount so a still-busy mount doesn't wedge
    # the script; the dirty bytes have already been snapshotted by then.
    if mountpoint -q "$MNT" 2>/dev/null; then
        umount -l "$MNT" 2>/dev/null || true
    fi
    if [[ -n "$LOOP" ]]; then
        losetup -d "$LOOP" 2>/dev/null || true
    fi
    rmdir "$MNT" 2>/dev/null || true
    rm -f "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

echo "==> creating 64 MiB ext4 image (block size ${BS}) at ${WORK}"
# 64 MiB gives the kernel room for a real journal and several transactions.
dd if=/dev/zero of="$WORK" bs=1M count=64 status=none
# -F: force on a regular file. The journal lands on inode 8 by default.
mkfs.ext4 -F -b "$BS" "$WORK" >/dev/null

echo "==> attaching loop device"
LOOP="$(losetup --find --show "$WORK")"
echo "    loop device: ${LOOP}"

echo "==> mounting (default data=ordered; the kernel journals metadata)"
mount "$LOOP" "$MNT"

echo "==> writing files/dirs so the kernel logs real transactions"
mkdir -p "$MNT/dir_a/dir_b"
for i in $(seq 1 32); do
    # A mix of small and larger files exercises descriptor + multi-block tags.
    head -c "$((4096 * (i % 8 + 1)))" /dev/urandom > "$MNT/dir_a/file_${i}.bin"
done
ln -s dir_a/file_1.bin "$MNT/symlink_1"
ln "$MNT/dir_a/file_2.bin" "$MNT/dir_a/dir_b/hardlink_2"

# Force the kernel to COMMIT the in-memory transactions to the on-disk log
# WITHOUT checkpointing them to their final locations: `sync` flushes the
# journal; we then snapshot before any unmount can checkpoint + clear it.
# (data=ordered: file data is written, metadata lives in the journal.)
echo "==> syncing so transactions are committed to the on-disk log"
sync

echo "==> snapshotting the raw image while still mounted (journal mid-flight)"
# Copy the underlying file's bytes NOW. Because the fs is still mounted and not
# unmounted, the superblock should still carry needs_recovery and the log should
# still hold un-checkpointed transactions.
cp --reflink=auto "$WORK" "$OUT" 2>/dev/null || cp "$WORK" "$OUT"

echo "==> lazy-unmounting the original (the snapshot is already taken)"
umount -l "$MNT" 2>/dev/null || true

echo
echo "==> VERIFY the captured fixture has a DIRTY journal before trusting it:"
echo "    dumpe2fs -h '${OUT}' | grep -iE 'Filesystem state|Journal'"
echo "      -> 'Filesystem state' should NOT be just 'clean'; look for"
echo "         'not clean with errors'/'recover' or a non-empty journal."
echo "    debugfs -R 'logdump -a' '${OUT}'"
echo "      -> should print 'Journal starts at block ...' and at least one"
echo "         '(descriptor block)' + '(commit block)'. An EMPTY dump means the"
echo "         kernel checkpointed everything; the image is NOT a dirty fixture."
echo
# Print a quick first-look so the operator sees the state immediately.
echo "==> dumpe2fs header (state + journal summary):"
dumpe2fs -h "$OUT" 2>/dev/null | grep -iE 'Filesystem state|needs_recovery|Journal' || true
echo
echo "==> logdump (first lines):"
debugfs -R "logdump -a" "$OUT" 2>/dev/null | head -20 || true

echo
echo "==> DONE. If VERIFY shows a dirty journal:"
echo "    1. mv '${OUT}' tests/fixtures/dirty/<name>.img"
echo "    2. EXT4_KERNEL_FIXTURES=1 cargo test --test journal_harness kernel_dirty -- --ignored"
