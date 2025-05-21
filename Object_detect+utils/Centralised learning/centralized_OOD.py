# coco_train_binary.py
#
# A minimal rewrite of coco_train.py that groups the 22 fine-grained
# classes into three super-classes:
#   0 – stand-still  (benches, poles, …)
#   1 – moving       (cars, bikes, …)
#   2 – background   (no labelled object in frame)
#
# ────────────────────────────────────────────────────────────────────────────
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import config1 as config1           # ← your existing config file

# ─── SUPER-CLASS MAPPING ────────────────────────────────────────────────────
# Convert original 0-based class IDs (0‥21) to 0/1/2
MOVING_CLASSES     = {1, 2, 4, 7, 9, 10, 17, 19}     # bicycle, bus, car, dog, motorcycle, person, train, truck
STANDSTILL_CLASSES = set(range(22)) - MOVING_CLASSES

LABEL_STANDSTILL = 0
LABEL_MOVING     = 1
LABEL_BACKGROUND = 2
NUM_CLASSES      = 3

# Tell the imported config that we now have three classes
config1.NUM_CLASSES = NUM_CLASSES

def map_to_binary(label_0):
    """Map a scalar tensor (0‥21) to 0, 1, or 2."""
    return tf.case(
        [
            (tf.reduce_any(tf.equal(label_0, list(MOVING_CLASSES))),
             lambda: tf.constant(LABEL_MOVING, tf.int32)),
            (tf.reduce_any(tf.equal(label_0, list(STANDSTILL_CLASSES))),
             lambda: tf.constant(LABEL_STANDSTILL, tf.int32)),
        ],
        default=lambda: tf.constant(LABEL_BACKGROUND, tf.int32)
    )

# ─── TFRECORD PARSING ───────────────────────────────────────────────────────
def parse_tfrecord(example_proto):
    feats = {
        'image/encoded':            tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, feats)

    # image → float32 [0,1]
    img = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    img = tf.image.resize(img, (config1.IMAGE_HEIGHT, config1.IMAGE_WIDTH))
    img = tf.cast(img, tf.float32) / 255.0

    # class label(s) → single binary label
    raw_labels = tf.sparse.to_dense(x['image/object/class/label']) - 1  # 1-based → 0-based
    label = tf.cond(
        tf.size(raw_labels) > 0,
        lambda: map_to_binary(raw_labels[0]),
        lambda: tf.constant(LABEL_BACKGROUND, dtype=tf.int32)
    )
    return img, label

# ─── DATA AUGMENTATION (unchanged) ──────────────────────────────────────────
def augment(img, label):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    return img, label

# ─── DATASET LOADING (unchanged) ────────────────────────────────────────────
def load_dataset(tfrecord_path, shuffle=True, batch_size=32, augment_data=False):
    ds = tf.data.TFRecordDataset(tfrecord_path)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ─── MODEL DEFINITION (same backbone, new Dense size) ───────────────────────
def create_model():
    inp = layers.Input((config1.IMAGE_HEIGHT, config1.IMAGE_WIDTH, 3))

    # Conv blocks
    x = layers.Conv2D(16, 3, padding='same')(inp)
    x = layers.BatchNormalization()(x);  x = layers.ReLU()(x);  x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x);  x = layers.ReLU()(x);  x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x);  x = layers.ReLU()(x);  x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x);  x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(NUM_CLASSES, activation='softmax', name='class_output')(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ─── TRAIN / VALID / TEST ───────────────────────────────────────────────────
if __name__ == '__main__':
    train_ds = load_dataset(
        config1.TRAIN_RECORD_PATH, shuffle=False, batch_size=8, augment_data=True
    )
    valid_ds = load_dataset(
        config1.VALID_RECORD_PATH, shuffle=False, batch_size=8, augment_data=False
    )
    test_ds  = load_dataset(
        config1.TEST_RECORD_PATH,  shuffle=False, batch_size=8, augment_data=False
    )

    model = create_model()
    model.summary()

    lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=20,
        callbacks=[lr_cb],
        verbose=1
    )

    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    save_path = os.path.join(config1.BASE_DIR, 'SA_classification_binary11.keras')
    model.save(save_path)
    print(f"Model saved to {save_path}")
