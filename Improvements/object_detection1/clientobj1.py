#!/usr/bin/env python3

import os
import time
import socket
import pickle
import struct
import platform
import psutil
import argparse

import tensorflow as tf
from tensorflow.keras import layers, models, metrics

import config1

# ─── ARGPARSE FOR CLIENT ID ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Federated learning client")
parser.add_argument(
    "--client_id", type=int, required=True,
    help="Client index for data split assignment (0-based)"
)
args = parser.parse_args()
client_id = args.client_id

# ─── DATA SPLIT SELECTION ────────────────────────────────────────────────────
splits_dir = os.path.join(config1.BASE_DIR, "OOD.v1i.tfrecord_splits")
split_names = sorted([
    d for d in os.listdir(splits_dir)
    if os.path.isdir(os.path.join(splits_dir, d)) and d.startswith("split_")
])
if not split_names:
    raise ValueError(f"No splits found in {splits_dir!r}")
selected_split = split_names[client_id % len(split_names)]
split_dir = os.path.join(splits_dir, selected_split)

TRAIN_RECORD_PATH = os.path.join(split_dir, "train", "train.tfrecord")
VALID_RECORD_PATH = os.path.join(split_dir, "valid", "valid.tfrecord")
TEST_RECORD_PATH  = os.path.join(split_dir, "test",  "test.tfrecord")

print(f"Using data split: {selected_split}")
print(f"  TRAIN_RECORD_PATH: {TRAIN_RECORD_PATH}")
print(f"  VALID_RECORD_PATH: {VALID_RECORD_PATH}")
print(f"  TEST_RECORD_PATH:  {TEST_RECORD_PATH}")

# ─── TF + MODEL SETUP ────────────────────────────────────────────────────────
# Force CPU for consistency
os.environ["CUDA_VISIBLE_DEVICES"]    = "-1"
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["TF_NUM_INTRAOP_THREADS"]  = "1"
os.environ["TF_NUM_INTEROP_THREADS"]  = "1"

def parse_tfrecord(example_proto):
    features = {
        'image/encoded':            tf.io.FixedLenFeature([],     tf.string),
        'image/height':             tf.io.FixedLenFeature([],     tf.int64),
        'image/width':              tf.io.FixedLenFeature([],     tf.int64),
        'image/object/bbox/xmin':   tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':   tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':   tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':   tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    ex = tf.io.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(ex['image/encoded'], channels=3)
    img = tf.image.resize(img, (config1.IMAGE_HEIGHT, config1.IMAGE_WIDTH)) / 255.0

    xmin   = tf.sparse.to_dense(ex['image/object/bbox/xmin'])
    xmax   = tf.sparse.to_dense(ex['image/object/bbox/xmax'])
    ymin   = tf.sparse.to_dense(ex['image/object/bbox/ymin'])
    ymax   = tf.sparse.to_dense(ex['image/object/bbox/ymax'])
    labels = tf.sparse.to_dense(ex['image/object/class/label']) - 1

    boxes  = tf.stack([ymin, xmin, ymax, xmax], axis=1)[:config1.NUM_BOXES]
    labels = labels[:config1.NUM_BOXES]
    boxes  = tf.pad(boxes,  [[0, config1.NUM_BOXES - tf.shape(boxes)[0]], [0,0]])
    labels = tf.pad(labels, [[0, config1.NUM_BOXES - tf.shape(labels)[0]]])
    return img, (boxes, labels)

def load_dataset(path):
    ds = tf.data.TFRecordDataset(path)
    ds = ds.shuffle(100).map(parse_tfrecord)
    ds = ds.padded_batch(1).prefetch(tf.data.AUTOTUNE)
    return ds

def create_model():
    inp = tf.keras.Input((config1.IMAGE_HEIGHT, config1.IMAGE_WIDTH, 3))
    x   = layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
    x   = layers.MaxPooling2D()(x)
    x   = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x   = layers.GlobalAveragePooling2D()(x)

    bbox = layers.Dense(config1.NUM_BOXES * 4, activation='sigmoid')(x)
    bbox = layers.Reshape((config1.NUM_BOXES, 4), name='bbox_output')(bbox)

    cls  = layers.Dense(config1.NUM_BOXES * config1.NUM_CLASSES, activation='softmax')(x)
    cls  = layers.Reshape((config1.NUM_BOXES, config1.NUM_CLASSES), name='class_output')(cls)

    model = models.Model(inp, [bbox, cls])
    model.compile(
        optimizer='adam',
        loss={
            'bbox_output':  'mse',
            'class_output': 'sparse_categorical_crossentropy'
        },
        metrics={'class_output': metrics.SparseCategoricalAccuracy(name='accuracy')}
    )
    return model

# ─── ENERGY‐METER SETUP ───────────────────────────────────────────────────────
meter = None
if platform.system() == "Linux":
    try:
        from pyJoules.energy_meter import EnergyMeter
        from pyJoules.device.rapl_device import RaplDevice
        domains = RaplDevice.available_domains()
        meter   = EnergyMeter(*domains)
    except Exception:
        meter = None

# ─── MAIN FEDERATED CLIENT WORKFLOW ──────────────────────────────────────────
if __name__ == "__main__":
    train_ds = load_dataset(TRAIN_RECORD_PATH)
    model    = create_model()
    proc     = psutil.Process()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((config1.SERVER_IP, config1.PORT))
    print("✅ Connected to server.")

    for rnd in range(config1.NUM_ROUNDS):
        print(f"\n🔁 Round {rnd+1}")

        # 1) Get global weights
        size_bytes = sock.recv(4)
        size       = struct.unpack(">I", size_bytes)[0]
        data       = b""
        while len(data) < size:
            data += sock.recv(4096)
        model.set_weights(pickle.loads(data))

        # 2) Start energy/time
        if meter:
            meter.start()
        cpu0 = sum(proc.cpu_times()[:2])
        t0   = time.time()

        # 3) Train
        hist = model.fit(train_ds,
                         epochs=config1.EPOCHS_PER_ROUND,
                         verbose=1)

        # 4) Stop & compute energy
        t1 = time.time()
        if meter:
            meter.stop()
            joules = sum(r.energy for r in meter.result)
        else:
            cpu1    = sum(proc.cpu_times()[:2])
            joules  = (cpu1 - cpu0) * config1.CPU_TDP
        duration = t1 - t0
        watts    = joules / duration if duration > 0 else 0.0

        # 5) Extract accuracy & loss
        # note: Keras records this under 'class_output_accuracy'
        hist_acc = hist.history.get('class_output_accuracy',
                  hist.history.get('accuracy', [None]))
        acc      = hist_acc[-1]

        loss     = hist.history.get('loss', [None])[-1]
        print(f"🔋 Energy: {joules:.1f} J over {duration:.1f}s → {watts:.1f} W")
        print(f"📈 Loss={loss:.4f}, Accuracy={(acc*100):.2f}%" if acc is not None else "Accuracy=N/A")

        # 6) Send weights + metrics
        out = {
            'weights':  model.get_weights(),
            'joules':   joules,
            'duration': duration,
            'accuracy': acc
        }
        payload = pickle.dumps(out)
        header  = struct.pack(">I", len(payload))
        sock.sendall(header + payload)

    sock.close()
    print("🚪 Disconnected.")
