#!/usr/bin/env python3
"""
Federated-learning client for 3-class moving / stand-still classification
— now with *in-training* power monitoring for CPU & GPU rails and communication steps.
"""

import os, time, socket, pickle, struct, argparse, psutil, statistics
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import config1 as config                      # MOVING_IDS, NUM_FINE_CLASSES, …

from jtop import jtop


# ────────────────────────── POWER SAMPLING HELPERS ─────────────────────────
def _snapshot_power(rail="POM_5V_CPU"):
    """
    Take a 1-second jtop snapshot and return the requested rail power in watts.
    Rail names on Jetson are usually:  POM_5V_CPU, POM_5V_GPU, POM_5V_IN, …
    """
    with jtop() as jetson:
        p = jetson.power
    try:
        mw = p["rail"][rail]["power"]
        return mw / 1000.0
    except Exception:
        print(f"⚠️  jtop rail structure changed → {p}")
        return float("nan")


class PowerMetric(callbacks.Callback):
    """
    Samples the specified rail power *during* training.
      • on_train_batch_begin: snapshot every `sample_every` batches
      • on_epoch_end       : print epoch mean & push to list
      • .mean_power        : round-level mean across epochs
    """
    def __init__(self, rail="POM_5V_GPU", sample_every=10):
        super().__init__()
        self.rail = rail
        self.sample_every = sample_every
        self._batch_counter = 0
        self._samples = []
        self._epoch_means = []

    def on_train_batch_begin(self, batch, logs=None):
        if self._batch_counter % self.sample_every == 0:
            self._samples.append(_snapshot_power(self.rail))
        self._batch_counter += 1

    def on_epoch_end(self, epoch, logs=None):
        if self._samples:
            epoch_mean = statistics.fmean(self._samples)
            self._epoch_means.append(epoch_mean)
            print(f"Epoch {epoch+1:02d}: mean {self.rail} = {epoch_mean:.2f} W "
                  f"from {len(self._samples)} samples")
            if logs is not None:
                logs[f"mean_{self.rail.lower()}"] = epoch_mean
        # reset for next epoch
        self._samples.clear()
        self._batch_counter = 0

    @property
    def mean_power(self):
        return statistics.fmean(self._epoch_means) if self._epoch_means else float("nan")

    def reset_round(self):
        self._epoch_means.clear()


# ──────────────────────────── DATA UTILITIES ───────────────────────────────
MOVING_CLASSES     = set(config.MOVING_IDS)
STANDSTILL_CLASSES = set(range(config.NUM_FINE_CLASSES)) - MOVING_CLASSES


def map_to_binary(label0):
    return tf.case(
        [
            (tf.reduce_any(tf.equal(label0, list(MOVING_CLASSES))),
             lambda: tf.constant(1, tf.int32)),              # moving
            (tf.reduce_any(tf.equal(label0, list(STANDSTILL_CLASSES))),
             lambda: tf.constant(0, tf.int32)),              # stand-still
        ],
        default=lambda: tf.constant(2, tf.int32)             # background
    )


def augment(img, label):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    return img, label


def parse_tfrecord(example_proto):
    feats = {
        "image/encoded":            tf.io.FixedLenFeature([], tf.string),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, feats)
    img = tf.image.decode_jpeg(x["image/encoded"], 3)
    img = tf.image.resize(img, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)) / 255.0

    raw = tf.sparse.to_dense(x["image/object/class/label"]) - 1
    label = tf.cond(
        tf.size(raw) > 0,
        lambda: map_to_binary(raw[0]),
        lambda: tf.constant(2, tf.int32)
    )
    return img, label


def load_dataset(path, shuffle, batch_size, augment_data):
    ds = tf.data.TFRecordDataset(path)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ───────────────────────────── MODEL DEFINITION ────────────────────────────
def create_model():
    inp = layers.Input((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    x = inp
    for f in [16, 32, 64]:
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        if f == 64:
            x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ────────────────────────────────── MAIN ────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_id", type=int, required=True)
    cid = ap.parse_args().client_id

    # ── data paths (adjusted to fixed split dir) ───────────────────────────
    BASE_DIR = "/home/jetson/Downloads/jettie"
    train_ds = load_dataset(os.path.join(BASE_DIR, "split_01", "train", "train.tfrecord"),
                            True,  config.BATCH_SIZE, True)
    valid_ds = load_dataset(os.path.join(BASE_DIR, "split_01", "valid", "valid.tfrecord"),
                            False, config.BATCH_SIZE, False)
    test_ds  = load_dataset(os.path.join(BASE_DIR, "split_01", "test",  "test.tfrecord"),
                            False, config.BATCH_SIZE, False)

    # ── force CPU (remove if you want GPU training!) ──────────────────────
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "-1",
        "OMP_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
    })

    model = create_model()
    sock  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((config.SERVER_IP, config.PORT))
    print("Connected to server.")

    lr_cb    = callbacks.ReduceLROnPlateau("val_accuracy", 0.5, 2,
                                           min_lr=1e-6, verbose=1)
    power_cb = PowerMetric(rail="POM_5V_CPU", sample_every=10)   # sample every 10 batches
    proc     = psutil.Process()

    for rnd in range(config.NUM_ROUNDS):
        print(f"\n─── Round {rnd+1} ─────────────────────────────────────────")

        # … receive global weights …
        size = struct.unpack(">I", sock.recv(4))[0]
        buf  = b""
        while len(buf) < size:
            buf += sock.recv(4096)
        model.set_weights(pickle.loads(buf))
        recv_power = _snapshot_power("POM_5V_CPU")
        print(f"After receiving weights: CPU rail = {recv_power:.2f} W")

        # … local training …
        cpu0 = sum(proc.cpu_times()[:2]); t0 = time.time()
        hist = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=config.EPOCHS_PER_ROUND,
            callbacks=[lr_cb, power_cb],
            verbose=1
        )
        t1   = time.time(); cpu1 = sum(proc.cpu_times()[:2])

        joules    = (cpu1 - cpu0) * config.CPU_TDP
        duration  = t1 - t0
        cpu_watt  = joules / duration if duration else 0.0
        n_samples = hist.params.get("samples", 0)
        loss, acc = model.evaluate(test_ds, verbose=0)
        print(f"Local test accuracy: {acc*100:.2f}%")

        # … send update …
        payload = pickle.dumps({
            "weights":     model.get_weights(),
            "joules":      joules,
            "duration":    duration,
            "accuracy":    hist.history["accuracy"][-1],
            "num_samples": n_samples,
        })
        sock.sendall(struct.pack(">I", len(payload)) + payload)
        send_power = _snapshot_power("POM_5V_CPU")
        print(f"After sending update:   CPU rail = {send_power:.2f} W")

        # … round summary …
        print("🔋  CPU energy: "
              f"{joules:.1f} J over {duration:.1f} s → {cpu_watt:.1f} W")
        print(f"⚡  Mean {power_cb.rail} during training: "
              f"{power_cb.mean_power:.2f} W")

        # prepare for next round
        power_cb.reset_round()

    sock.close(); print("🚪 Disconnected.")
