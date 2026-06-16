# Authentic kernel dirty-journal fixtures (Tier 3)

This directory holds `*.img` ext4 images whose jbd2 journal is **dirty** — i.e.
they carry committed-but-un-checkpointed transactions written by a **real Linux
kernel** (CSUM_V3, multi-transaction, real escape cases). They drive the
`kernel_dirty_fixtures_recover_clean` test in `tests/journal_harness.rs`, which
replays each one through `Ext4::open_and_recover` and then asserts the result is
`e2fsck`-clean.

## These fixtures are NOT checked in

The `*.img` files are intentionally **git-ignored** (see the root `.gitignore`):
they are large binary blobs and can only be produced by a privileged run. Only
this `README.md` and an empty `.gitkeep` are tracked, so the directory exists in
a fresh checkout.

## Generating fixtures

Capturing an authentic dirty journal requires root (loop-mount a real image,
write into it, snapshot the raw device while the journal is mid-flight). The dev
sandbox is unprivileged and cannot do this. Use the helper script on a machine
where you have root:

```sh
sudo scripts/make-dirty-fixture.sh /tmp/fixture.img 4096
```

The script prints how to **verify** the captured image really has a dirty
journal (`dumpe2fs -h` / `debugfs -R "logdump -a"`). Only once verified, place
it here:

```sh
mv /tmp/fixture.img tests/fixtures/dirty/kernel_4k.img
```

## Running the test

The replay test is `#[ignore]`d and additionally gated on an env var, so it never
runs (and never fails on a missing fixture) in normal `cargo test`:

```sh
EXT4_KERNEL_FIXTURES=1 cargo test --test journal_harness kernel_dirty -- --ignored
```

It iterates every `*.img` in this directory, copies each to `target/` (recovery
mutates the image), recovers it, and asserts `e2fsck` reports it clean.
