"""
evaluate_binary.py
──────────────────
Evaluate a 3-class Keras model (stand-still / moving / background)
on a TFRecord test set that still contains the original 22 (or 30) labels.

Usage
─────
python evaluate_binary.py  \
    path/to/classification_binary_trans1.keras \
    path/to/TEST.tfrecord           \
    --batch_size 16
"""
import os
import argparse
import tensorflow as tf
import config1 as cfg           # defines IMAGE_HEIGHT/IMAGE_WIDTH, NUM_CLASSES is now 3

# ── SUPER-CLASS MAPPING (0-based IDs!) ──────────────────────────────────────
# *** Edit MOVING_CLASSES to match the dataset you trained on ***
MOVING_CLASSES = {
   0, 3, 6, 7, 8, 9, 16, 18, 27     # bicycle, bus, car, dog, motorcycle, person, train, truck
}
STANDSTILL_CLASSES = set(range(30)) - MOVING_CLASSES   # 0-21 fine-grained IDs

LABEL_STANDSTILL = 0
LABEL_MOVING     = 1
LABEL_BACKGROUND = 2

# ── TFRecord parsing → 3-class label ────────────────────────────────────────
def map_to_binary(label0):
    """Return 0 (stand-still), 1 (moving) or 2 (background)."""
    return tf.case(
        [
            (tf.reduce_any(tf.equal(label0, list(MOVING_CLASSES))),
             lambda: tf.constant(LABEL_MOVING, tf.int32)),
            (tf.reduce_any(tf.equal(label0, list(STANDSTILL_CLASSES))),
             lambda: tf.constant(LABEL_STANDSTILL, tf.int32)),
        ],
        default=lambda: tf.constant(LABEL_BACKGROUND, tf.int32)
    )

def parse_tfrecord(example_proto):
    feats = {
        'image/encoded':            tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, feats)

    img = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    img = tf.image.resize(img, (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
    img = tf.cast(img, tf.float32) / 255.0

    raw = tf.sparse.to_dense(x['image/object/class/label']) - 1     # 1-based → 0-based
    label = tf.cond(
        tf.size(raw) > 0,
        lambda: map_to_binary(raw[0]),
        lambda: tf.constant(LABEL_BACKGROUND, dtype=tf.int32)       # empty → background
    )
    return img, label

# ── Dataset loader ─────────────────────────────────────────────────────────
def load_dataset(tfrecord_path, batch_size=32):
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ── Evaluation entry point ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a 3-class moving/stand-still Keras model on TFRecords')
    parser.add_argument('model_path',   type=str, help='.keras model file')
    parser.add_argument('test_record',  type=str, help='TFRecord test file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    # Load model (no need to re-compile metrics, but harmless if we do)
    model = tf.keras.models.load_model(args.model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Dataset
    test_ds = load_dataset(args.test_record, batch_size=args.batch_size)

    # Evaluate
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest loss     : {loss:.4f}")
    print(f"Test accuracy : {acc * 100:.2f}%")

if __name__ == '__main__':
    main()
