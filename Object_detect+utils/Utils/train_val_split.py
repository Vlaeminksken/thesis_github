import tensorflow as tf
import os

# ─── SETTINGS ───────────────────────────────────────────────────────────────
merged_tfrecord_path = "merged_dataset.tfrecord"  # input file
train_output_path = "train.tfrecord"
val_output_path   = "val.tfrecord"
split_ratio = 0.8  # 80% train, 20% val

# ─── LOAD & BUFFER RECORDS ──────────────────────────────────────────────────
raw_dataset = tf.data.TFRecordDataset(merged_tfrecord_path)
raw_dataset = list(raw_dataset)  # materialize to memory for shuffling

# ─── SHUFFLE & SPLIT ────────────────────────────────────────────────────────
total_count = len(raw_dataset)
split_index = int(total_count * split_ratio)

# Optional: shuffle before splitting
import random
random.shuffle(raw_dataset)

train_records = raw_dataset[:split_index]
val_records   = raw_dataset[split_index:]

# ─── WRITE TFRECORDS ────────────────────────────────────────────────────────
with tf.io.TFRecordWriter(train_output_path) as train_writer:
    for record in train_records:
        train_writer.write(record.numpy())

with tf.io.TFRecordWriter(val_output_path) as val_writer:
    for record in val_records:
        val_writer.write(record.numpy())

print(f"Total samples: {total_count}")
print(f"Train samples: {len(train_records)} → {train_output_path}")
print(f"Val samples:   {len(val_records)} → {val_output_path}")
