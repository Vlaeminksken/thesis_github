#!/usr/bin/env python3
"""Federated-learning client for 3-class moving / stand-still classification."""
import os, time, socket, pickle, struct, argparse, psutil
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import config1 as config            # make sure this file has MOVING_IDS & NUM_FINE_CLASSES

# ─── SUPER-CLASS MAPPING ────────────────────────────────────────────────────
MOVING_CLASSES     = set(config.MOVING_IDS)
STANDSTILL_CLASSES = set(range(config.NUM_FINE_CLASSES)) - MOVING_CLASSES

def map_to_binary(label0):
    return tf.case(
        [
            (tf.reduce_any(tf.equal(label0, list(MOVING_CLASSES))),
             lambda: tf.constant(1, tf.int32)),     # moving
            (tf.reduce_any(tf.equal(label0, list(STANDSTILL_CLASSES))),
             lambda: tf.constant(0, tf.int32)),     # stand-still
        ],
        default=lambda: tf.constant(2, tf.int32)     # background
    )

# ─── DATA AUGMENTATION & TFRecord PARSE ────────────────────────────────────
def augment(img, label):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    return img, label

def parse_tfrecord(example_proto):
    feats = {
        'image/encoded':            tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, feats)
    img = tf.image.decode_jpeg(x['image/encoded'], 3)
    img = tf.image.resize(img, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)) / 255.0

    raw = tf.sparse.to_dense(x['image/object/class/label']) - 1  # 1->0 base
    label = tf.cond(
        tf.size(raw) > 0,
        lambda: map_to_binary(raw[0]),
        lambda: tf.constant(2, tf.int32)                          # background
    )
    return img, label

def load_dataset(path, shuffle, batch_size, augment_data):
    ds = tf.data.TFRecordDataset(path)
    if shuffle: ds = ds.shuffle(1000)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data: ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ─── MODEL DEFINITION (3-class head) ───────────────────────────────────────
def create_model():
    inp = layers.Input((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    x = inp
    for f in [16, 32, 64]:
        x = layers.Conv2D(f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x); x = layers.MaxPooling2D()(x)
        if f == 64: x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.5)(x)
    out = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)   # now 3
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

# ─── CLIENT MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--client_id", type=int, required=True)
    cid = ap.parse_args().client_id

    split_dirs = sorted(d for d in os.listdir(config.SPLITS_DIR) if d.startswith("split_"))
    split = split_dirs[cid % len(split_dirs)]
    base  = os.path.join(config.SPLITS_DIR, split)
    train_ds = load_dataset(os.path.join(base, "train", "train.tfrecord"),
                            True,  config.BATCH_SIZE, True)
    valid_ds = load_dataset(os.path.join(base, "valid", "valid.tfrecord"),
                            False, config.BATCH_SIZE, False)
    test_ds  = load_dataset(os.path.join(base, "test",  "test.tfrecord"),
                            False, config.BATCH_SIZE, False)

    # CPU-only
    os.environ.update({"CUDA_VISIBLE_DEVICES": "-1",
                       "OMP_NUM_THREADS": "1",
                       "TF_NUM_INTRAOP_THREADS": "1",
                       "TF_NUM_INTEROP_THREADS": "1"})

    model = create_model()
    sock  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((config.SERVER_IP, config.PORT))
    print("✅ Connected to server.")

    lr_cb = callbacks.ReduceLROnPlateau("val_accuracy", 0.5, 2, min_lr=1e-6, verbose=1)
    proc  = psutil.Process()

    for rnd in range(config.NUM_ROUNDS):
        print(f"\n🔁 Round {rnd+1}")

        # receive global weights
        size = struct.unpack(">I", sock.recv(4))[0]
        buf  = b""
        while len(buf) < size:
            buf += sock.recv(4096)
        model.set_weights(pickle.loads(buf))

        # local training
        cpu0 = sum(proc.cpu_times()[:2]); t0 = time.time()
        hist = model.fit(train_ds, validation_data=valid_ds,
                         epochs=config.EPOCHS_PER_ROUND,
                         callbacks=[lr_cb], verbose=1)
        t1   = time.time(); cpu1 = sum(proc.cpu_times()[:2])

        joules   = (cpu1 - cpu0) * config.CPU_TDP
        duration = t1 - t0
        watts    = joules / duration if duration else 0.0
        n_samples = hist.params.get("samples", 0)

        loss, acc = model.evaluate(test_ds, verbose=0)
        print(f"🔎 Local Test Accuracy: {acc*100:.2f}%")

        payload = pickle.dumps({
            "weights":     model.get_weights(),
            "joules":      joules,
            "duration":    duration,
            "accuracy":    hist.history["accuracy"][-1],
            "num_samples": n_samples,
        })
        sock.sendall(struct.pack(">I", len(payload)) + payload)
        print(f"🔋 Energy: {joules:.1f} J over {duration:.1f} s → {watts:.1f} W")

    sock.close(); print("🚪 Disconnected.")
