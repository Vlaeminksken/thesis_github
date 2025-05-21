import os
import tensorflow as tf

def split_tfrecord(
    input_path: str,
    output_dir: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    shuffle_buffer: int = 10000,
    seed: int = 42,
):
    """
    Splits a single TFRecord into train/val/test shards.

    Args:
        input_path:   Path to the source .tfrecord file.
        output_dir:   Directory where train/, val/, test/ subfolders will go.
        train_frac:   Fraction of examples for training.
        val_frac:     Fraction for validation.
        test_frac:    Fraction for testing.
        shuffle_buffer: Buffer size for shuffling.
        seed:         Random seed for reproducibility.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Splits must sum to 1."

    # 1) Read and count total examples
    raw_ds = tf.data.TFRecordDataset(input_path)
    total = 0
    for _ in raw_ds:
        total += 1
    print(f"Total examples found: {total}")

    # 2) Shuffle and repartition
    ds = tf.data.TFRecordDataset(input_path)
    ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=False)

    n_train = int(total * train_frac)
    n_val   = int(total * val_frac)
    n_test  = total - n_train - n_val

    train_ds = ds.take(n_train)
    rest     = ds.skip(n_train)
    val_ds   = rest.take(n_val)
    test_ds  = rest.skip(n_val)

    # 3) Prepare output dirs
    for split in ("train", "val", "test"):
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    # 4) Write out
    writers = {
        "train": tf.io.TFRecordWriter(os.path.join(output_dir, "train", "data.tfrecord")),
        "val":   tf.io.TFRecordWriter(os.path.join(output_dir, "val",   "data.tfrecord")),
        "test":  tf.io.TFRecordWriter(os.path.join(output_dir, "test",  "data.tfrecord")),
    }

    for example in train_ds:
        writers["train"].write(example.numpy())
    for example in val_ds:
        writers["val"].write(example.numpy())
    for example in test_ds:
        writers["test"].write(example.numpy())

    for w in writers.values():
        w.close()

    print(f"Wrote {n_train} train, {n_val} val, {n_test} test examples to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split a TFRecord into train/val/test")
    parser.add_argument("input",   help="Input .tfrecord file")
    parser.add_argument("output",  help="Output directory")
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction")
    parser.add_argument("--val",   type=float, default=0.1, help="Val fraction")
    parser.add_argument("--test",  type=float, default=0.1, help="Test fraction")
    parser.add_argument("--buffer",type=int,   default=10000, help="Shuffle buffer size")
    parser.add_argument("--seed",  type=int,   default=42,   help="Random seed")
    args = parser.parse_args()

    split_tfrecord(
        input_path=args.input,
        output_dir=args.output,
        train_frac=args.train,
        val_frac=args.val,
        test_frac=args.test,
        shuffle_buffer=args.buffer,
        seed=args.seed,
    )
