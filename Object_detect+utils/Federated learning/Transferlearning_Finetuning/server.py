#!/usr/bin/env python3
import socket, pickle, struct, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import config1 as cfg                        # MUST provide MOVING_IDS, NUM_FINE_CLASSES, NUM_CLASSES

# ─── SUPER-CLASS MAPPING ────────────────────────────────────────────────────
MOVING   = set(cfg.MOVING_IDS)
STATIC   = set(range(cfg.NUM_FINE_CLASSES)) - MOVING

def to_3class(label0):
    return tf.case(
        [
            (tf.reduce_any(tf.equal(label0, list(MOVING))),  lambda: tf.constant(1, tf.int32)),
            (tf.reduce_any(tf.equal(label0, list(STATIC))), lambda: tf.constant(0, tf.int32)),
        ],
        default=lambda: tf.constant(2, tf.int32)   # background
    )

# ─── PARSE TFRecord ────────────────────────────────────────────────────────
def parse_tfrecord(proto):
    feats = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
             'image/object/class/label': tf.io.VarLenFeature(tf.int64)}
    x = tf.io.parse_single_example(proto, feats)

    img   = tf.image.decode_jpeg(x['image/encoded'], 3)
    img   = tf.image.resize(img, (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)) / 255.0
    raw   = tf.sparse.to_dense(x['image/object/class/label']) - 1
    label = tf.cond(tf.size(raw) > 0, lambda: to_3class(raw[0]),
                    lambda: tf.constant(2, tf.int32))
    return img, label

def load_test(path, bs=1):
    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

# ─── BACKBONE & HEAD BUILDERS (unchanged except Dense size) ───────────────
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

def build_transfer_model(pretrained_path, num_unfreeze_convs=2):
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
    out = layers.Dense(cfg.NUM_CLASSES, activation='softmax')(x)
    model = models.Model(backbone.input, out)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_finetune():
    b = create_backbone(); b.load_weights(cfg.PRETRAINED_PATH, by_name=True); b.trainable = True
    out = layers.Dense(cfg.NUM_CLASSES, activation='softmax')(b.output)
    m   = models.Model(b.input, out)
    m.compile(optimizers.Adam(1e-4), 'sparse_categorical_crossentropy', ['accuracy'])
    return m

# ─── SERVER MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_transfer_model(cfg.PRETRAINED_PATH, 2) if cfg.MODE.lower() == 'transfer' else build_finetune()
    test_ds = load_test(cfg.TEST_RECORD_PATH)

    s = socket.socket(); s.bind((cfg.SERVER_IP, cfg.PORT)); s.listen(cfg.CLIENTS_COUNT)
    print("📡 Server ready.")
    conns = [s.accept()[0] for _ in range(cfg.CLIENTS_COUNT)]

    for rnd in range(cfg.NUM_ROUNDS):
        print(f"\n🔁 Round {rnd+1}")
        blob = pickle.dumps(model.get_weights()); hdr = struct.pack(">I", len(blob))
        for c in conns: c.sendall(hdr+blob)

        updates, accs = [], []
        for cid, c in enumerate(conns):
            try:
                sz = struct.unpack(">I", c.recv(4))[0]
                buf = b"" 
                while len(buf) < sz:
                    buf += c.recv(4096)
                rec = pickle.loads(buf)
                updates.append(rec["weights"]);  accs.append(rec["accuracy"])
            except Exception as e:
                print(f"⚠️ client {cid} failed:", e)

        if updates:
            w = np.power(accs, 2.0);  w /= w.sum()
            new = [sum(wi * wgt for wi, wgt in zip(layer, w)) for layer in zip(*updates)]
            model.set_weights(new)
            loss, acc = model.evaluate(test_ds, verbose=0)
            print(f"📊 Global acc {acc*100:.2f}%")

    model.save(f"fl_{cfg.MODE}_3class.keras")
    for c in conns: c.close();  s.close()
