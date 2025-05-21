# coco_transfer_binary.py  – 22-label backbone → 3-class head
# ────────────────────────────────────────────────────────────────────────────
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import config1 as cfg                    # must define IMAGE/HEIGHT/WIDTH, paths, BASE_DIR

# ─── SUPER-CLASS MAPPING (use 0-based IDs!) ─────────────────────────────────
MOVING_CLASSES = {
    1,   # bicycle
    2,   # bus
    4,   # car
    7,   # dog
    9,   # motorcycle
    10,  # person
    17,  # train
    19,  # truck
}
STANDSTILL_CLASSES = set(range(22)) - MOVING_CLASSES   # 22 object IDs → 0-21

LABEL_STANDSTILL = 0
LABEL_MOVING     = 1
LABEL_BACKGROUND = 2
NUM_CLASSES      = 3

# Tell the rest of the code (and anything else that imports cfg) we have 3 classes
cfg.NUM_CLASSES = NUM_CLASSES

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

# ─── TFRECORD PARSING ───────────────────────────────────────────────────────
def parse_tfrecord(example_proto):
    feats = {
        'image/encoded':            tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, feats)

    # image → float32 [0,1]
    img = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    img = tf.image.resize(img, (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
    img = tf.cast(img, tf.float32) / 255.0

    # label(s) → single 0/1/2
    raw = tf.sparse.to_dense(x['image/object/class/label']) - 1  # 1-based → 0-based
    label = tf.cond(
        tf.size(raw) > 0,
        lambda: map_to_binary(raw[0]),
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

# ─── BACKBONE (unchanged) ───────────────────────────────────────────────────
def create_backbone():
    inp = layers.Input((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3))
    x = inp
    for filters in [16, 32, 64]:
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        if filters == 64:
            x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    return models.Model(inp, x, name='backbone')

# ─── TRANSFER vs. FINETUNE HEADS (Dense size = 3) ──────────────────────────
def build_transfer_model(pretrained_path, num_unfreeze_convs=4):
    backbone = create_backbone()
    backbone.load_weights(pretrained_path, by_name=True)

    # 1) freeze everything
    for layer in backbone.layers:
        layer.trainable = False

    # 2) unfreeze last N conv layers (+ their BN) to adapt
    conv_layers = [l for l in backbone.layers if isinstance(l, layers.Conv2D)]
    for conv in conv_layers[-num_unfreeze_convs:]:
        conv.trainable = True
        bn_name = conv.name.replace('conv2d', 'batch_normalization')
        try:
            backbone.get_layer(bn_name).trainable = True
        except KeyError:
            pass

    # fresh 3-class head
    x = backbone.output
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(backbone.input, out)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_finetune_model(pretrained_path):
    backbone = create_backbone()
    backbone.load_weights(pretrained_path, by_name=True)
    backbone.trainable = True

    out = layers.Dense(NUM_CLASSES, activation='softmax')(backbone.output)
    model = models.Model(backbone.input, out)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ─── TRAIN / VALID / TEST (unchanged) ───────────────────────────────────────
if __name__ == '__main__':
    pretrained = r"C:\Users\joran\thesis_github\SA_classification_binary_VIS1.keras"
    if not os.path.exists(pretrained):
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained}")

    train_ds = load_dataset(cfg.TRAIN_RECORD_PATH, shuffle=True,  batch_size=8, augment_data=True)
    valid_ds = load_dataset(cfg.VALID_RECORD_PATH, shuffle=False, batch_size=8, augment_data=False)
    test_ds  = load_dataset(cfg.TEST_RECORD_PATH,  shuffle=False, batch_size=8, augment_data=False)

    # pick one:
    model = build_transfer_model(pretrained, num_unfreeze_convs=4)
    #model = build_finetune_model(pretrained)

    model.summary()

    lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )
    ckpt_cb = callbacks.ModelCheckpoint(
        filepath='best_weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=10,
        callbacks=[lr_cb, ckpt_cb],
        verbose=1
    )

    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    save_path = os.path.join(cfg.BASE_DIR, 'classification_binary_fine1.keras')
    model.save(save_path)
    print(f"✅ Model saved to {save_path}")
