#!/usr/bin/env python3
"""Federated-learning server with accuracy-weighted FedAvg (3-class output)."""
import socket, pickle, struct, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import config1 as config          # must include MOVING_IDS & NUM_FINE_CLASSES

# ─── SUPER-CLASS MAPPING ────────────────────────────────────────────────────
MOVING_CLASSES     = set(config.MOVING_IDS)
STANDSTILL_CLASSES = set(range(config.NUM_FINE_CLASSES)) - MOVING_CLASSES

def map_to_binary(label0):
    return tf.case(
        [
            (tf.reduce_any(tf.equal(label0, list(MOVING_CLASSES))),
             lambda: tf.constant(1, tf.int32)),
            (tf.reduce_any(tf.equal(label0, list(STANDSTILL_CLASSES))),
             lambda: tf.constant(0, tf.int32)),
        ],
        default=lambda: tf.constant(2, tf.int32)
    )

# ─── TFRecord PARSING & TEST DATA ───────────────────────────────────────────
def parse_tfrecord(example_proto):
    feats = {
        'image/encoded':            tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, feats)
    img = tf.image.decode_jpeg(x['image/encoded'], 3)
    img = tf.image.resize(img, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)) / 255.0

    raw = tf.sparse.to_dense(x['image/object/class/label']) - 1
    label = tf.cond(
        tf.size(raw) > 0,
        lambda: map_to_binary(raw[0]),
        lambda: tf.constant(2, tf.int32)
    )
    return img, label

def load_test_dataset(path, batch_size=1):
    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
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

# ─── SERVER MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    server_model = create_model()
    test_ds = load_test_dataset(config.TEST_RECORD_PATH, batch_size=1)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((config.SERVER_IP, config.PORT)); srv.listen(config.CLIENTS_COUNT)
    print("📡 Server listening…")

    conns = []
    for i in range(config.CLIENTS_COUNT):
        conn, addr = srv.accept(); conn.settimeout(100000)
        print(f"🔗 Client {i} connected from {addr}"); conns.append(conn)

    for rnd in range(config.NUM_ROUNDS):
        print(f"\n🔁 Round {rnd+1}")

        # 1 – broadcast global weights
        blob = pickle.dumps(server_model.get_weights())
        hdr  = struct.pack(">I", len(blob))
        for c in conns: c.sendall(hdr + blob)

        # 2 – collect client updates
        updates, metrics = [], []   # metrics: (cid, joules, sec, power, acc)
        for cid, c in enumerate(conns):
            try:
                sz = struct.unpack(">I", c.recv(4))[0]
                buf = b"" 
                while len(buf) < sz:
                    buf += c.recv(4096)
                rec = pickle.loads(buf)
                updates.append(rec["weights"])
                j, d = rec["joules"], rec["duration"]
                pwr  = j / d if d else 0.0
                metrics.append((cid, j, d, pwr, rec["accuracy"]))
            except Exception as e:
                print(f"⚠️  Client {cid} failed: {e}")

        # 3 – accuracy-weighted FedAvg
        if updates:
            accs  = np.array([m[4] for m in metrics], np.float32)
            beta  = 1.0
            wts   = np.power(accs, beta); wts /= wts.sum()

            new_w = []
            for layer in zip(*updates):
                layer_avg = sum(w * wgt for w, wgt in zip(layer, wts))
                new_w.append(layer_avg)
            server_model.set_weights(new_w)
            print(f"✅ FedAvg (β={beta}) applied.")

            for cid, j, d, p, a in metrics:
                print(f"   • Client {cid}: {j:.1f} J / {d:.1f} s → {p:.1f} W, acc={a*100:.2f}%")

            loss, acc = server_model.evaluate(test_ds, verbose=0)
            print(f"📊 Global Test Accuracy: {acc*100:.2f}%  (loss {loss:.4f})")
        else:
            print("⚠️  No updates received this round.")

    server_model.save("fl_classification_3class1.keras")
    print("💾 Global model saved.")
    for c in conns: c.close()
    srv.close()
