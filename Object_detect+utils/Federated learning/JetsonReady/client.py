#!/usr/bin/env python3
import os, time, socket, pickle, struct, psutil, argparse
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import config1 as cfg                       # MUST provide MOVING_IDS, NUM_FINE_CLASSES, NUM_CLASSES

# ─── SUPER-CLASS MAPPING ────────────────────────────────────────────────────
MOVING   = set(cfg.MOVING_IDS)
STATIC   = set(range(cfg.NUM_FINE_CLASSES)) - MOVING
def to_3class(label0):
    return tf.case(
        [
            (tf.reduce_any(tf.equal(label0, list(MOVING))),  lambda: tf.constant(1, tf.int32)),
            (tf.reduce_any(tf.equal(label0, list(STATIC))),  lambda: tf.constant(0, tf.int32)),
        ],
        default=lambda: tf.constant(2, tf.int32)
    )

# ─── PARSE TFRecord ────────────────────────────────────────────────────────
def parse(proto):
    feats = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
             'image/object/class/label': tf.io.VarLenFeature(tf.int64)}
    x = tf.io.parse_single_example(proto, feats)
    img = tf.image.decode_jpeg(x['image/encoded'], 3)
    img = tf.image.resize(img, (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)) / 255.0
    raw = tf.sparse.to_dense(x['image/object/class/label']) - 1
    label = tf.cond(tf.size(raw)>0, lambda: to_3class(raw[0]),
                    lambda: tf.constant(2, tf.int32))
    return img, label

def dataset(path, shuf, bs, aug):
    ds = tf.data.TFRecordDataset(path)
    if shuf: ds = ds.shuffle(1000)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    if aug:
        ds = ds.map(lambda i,l: (tf.image.random_flip_left_right(i), l),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

# ─── MODEL (identical backbone, 3-class head) ──────────────────────────────
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
    out = layers.Dense(cfg.NUM_CLASSES,activation='softmax')(b.output)
    m   = models.Model(b.input,out)
    m.compile(optimizers.Adam(1e-4),'sparse_categorical_crossentropy',['accuracy'])
    return m

# ─── CLIENT MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--client_id", type=int, required=True)
    cid = ap.parse_args().client_id

    split_dirs = sorted(d for d in os.listdir(cfg.SPLITS_DIR) if d.startswith("split_"))
    base = os.path.join(cfg.SPLITS_DIR, split_dirs[cid % len(split_dirs)])
    train = dataset(os.path.join(base,"train","train.tfrecord"), True,  cfg.BATCH_SIZE, True)
    valid = dataset(os.path.join(base,"valid","valid.tfrecord"), False, cfg.BATCH_SIZE, False)
    test  = dataset(os.path.join(base,"test","test.tfrecord"),  False, cfg.BATCH_SIZE, False)

    os.environ.update({"CUDA_VISIBLE_DEVICES":"-1","OMP_NUM_THREADS":"1",
                       "TF_NUM_INTRAOP_THREADS":"1","TF_NUM_INTEROP_THREADS":"1"})

    model = build_transfer_model(cfg.PRETRAINED_PATH,2) if cfg.MODE.lower()=="transfer" else build_finetune()

    s = socket.socket(); s.connect((cfg.SERVER_IP, cfg.PORT)); print("✅ connected")

    lr_cb = callbacks.ReduceLROnPlateau("val_accuracy",0.5,2,min_lr=1e-6,verbose=1)
    proc  = psutil.Process()

    for rnd in range(cfg.NUM_ROUNDS):
        print(f"\n🔁 Round {rnd+1}")
        sz = struct.unpack(">I", s.recv(4))[0]
        buf = b""
        while len(buf) < sz:
            buf += s.recv(4096)
        model.set_weights(pickle.loads(buf))

        cpu0 = sum(proc.cpu_times()[:2]); t0 = time.time()
        hist = model.fit(train, validation_data=valid,
                         epochs=cfg.EPOCHS_PER_ROUND,
                         callbacks=[lr_cb], verbose=1)
        t1   = time.time(); cpu1 = sum(proc.cpu_times()[:2])

        joules = (cpu1 - cpu0) * cfg.CPU_TDP
        out = pickle.dumps({
            "weights":   model.get_weights(),
            "joules":    joules,
            "duration":  t1-t0,
            "accuracy":  hist.history["accuracy"][-1],
        })
        s.sendall(struct.pack(">I", len(out)) + out)

        loss, acc = model.evaluate(test, verbose=0)
        print(f"🔎 local acc {acc*100:.2f}%  🔋 {joules:.1f} J")

    s.close(); print("🚪 bye")
