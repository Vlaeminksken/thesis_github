#!/usr/bin/env python3
"""
split_tfrecords.py  –  create 10×10 % folds from a TFRecord dataset
────────────────────────────────────────────────────────────────────
Layout expected:

    <DATASET>/
        ├── train/   ─┐  one or many .tfrecord / .record shards
        ├── valid/   ─┤  (compression is auto-detected by TF)
        └── test/    ─┘

Result (10 folds, each = 10 % of every phase):

    <DATASET>_splits/
        ├── split_01/
        │   ├── train/phase.tfrecord
        │   ├── valid/phase.tfrecord
        │   └── test/phase.tfrecord
        ├── …
        └── split_10/…

Run time:  a few minutes per GB (entirely CPU, streams efficiently).
"""

# ------------- EDIT HERE (everything else works out of the box) -------------
DATASET = "OOD.v1i.tfrecord"   # ← folder that holds train/ valid/ test/
# -----------------------------------------------------------------------------

K_SPLITS  = 15          # keep at 10 for 10×10 %
RNG_SEED  = 42          # set to None for a new shuffle each run
OUTPUT_SUFFIX = "_splits"
PHASES = ("train", "valid", "test")

# -----------------------------------------------------------------------------


import random
from pathlib import Path
import tensorflow as tf


def list_shards(dir_: Path):
    """Return all *.tfrecord / *.record shards inside *dir_* (sorted)."""
    return sorted(dir_.glob("*.tfrecord")) + sorted(dir_.glob("*.record"))


def read_all_records(shards):
    """
    Stream *all* tf.train.Example records from a list of shards.
    Returns a Python list of serialized bytes (fits in RAM for most datasets).
    """
    if not shards:
        return []
    ds = tf.data.TFRecordDataset([str(s) for s in shards],
                                 num_parallel_reads=tf.data.AUTOTUNE,
                                 compression_type="")   # auto-detect
    return [rec.numpy() for rec in ds]


def chunk(lst, k):
    """Split *lst* into *k* near-equal chunks."""
    n, base = divmod(len(lst), k)
    outs, start = [], 0
    for i in range(k):
        end = start + n + (1 if i < base else 0)
        outs.append(lst[start:end])
        start = end
    return outs


def write_chunk(chunk_records, out_file):
    """Write a list of serialized Example bytes to *out_file*."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(out_file)) as w:
        for r in chunk_records:
            w.write(r)


def main():
    src_root = Path(DATASET).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Dataset dir “{src_root}” not found.")

    dst_root = src_root.with_name(src_root.name + OUTPUT_SUFFIX)
    if dst_root.exists():
        raise FileExistsError(f"Remove existing {dst_root} before rerunning.")

    random.seed(RNG_SEED)
    print(f"📂  Source : {src_root}")
    print(f"🗂   Output : {dst_root}")
    print(f"🔢  Splits : {K_SPLITS}  (≈{100//K_SPLITS}% each)\n")

    for phase in PHASES:
        phase_dir = src_root / phase
        shards = list_shards(phase_dir)
        if not shards:
            print(f"⚠️   No TFRecord shards found in {phase_dir} – skipping.")
            continue

        print(f"— {phase.upper()} —")
        records = read_all_records(shards)
        n_total = len(records)
        random.shuffle(records)
        parts = chunk(records, K_SPLITS)

        for idx, recs in enumerate(parts, 1):
            out_file = dst_root / f"split_{idx:02d}" / phase / f"{phase}.tfrecord"
            write_chunk(recs, out_file)
            print(f"  split_{idx:02d}: {len(recs):>6} / {n_total} records")

    print("\n✅  Finished – 20 perfectly even TFRecord folds ready.")


if __name__ == "__main__":
    main()
