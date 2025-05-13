# server.py

import socket
import pickle
import struct
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
import config2 as config1

def parse_tfrecord(example_proto):
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/height':  tf.io.FixedLenFeature([], tf.int64),
        'image/width':   tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
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

    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
    boxes = boxes[:config1.NUM_BOXES]
    labels = labels[:config1.NUM_BOXES]

    boxes = tf.pad(boxes,  [[0, config1.NUM_BOXES - tf.shape(boxes)[0]], [0, 0]])
    labels = tf.pad(labels, [[0, config1.NUM_BOXES - tf.shape(labels)[0]]])

    return img, (boxes, labels)

def load_test_dataset(path):
    return (
        tf.data.TFRecordDataset(path)
        .map(parse_tfrecord)
        .padded_batch(1)
        .prefetch(tf.data.AUTOTUNE)
    )

def create_model():
    inp = layers.Input((config1.IMAGE_HEIGHT, config1.IMAGE_WIDTH, 3))
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
            'bbox_output': 'mse',
            'class_output': 'sparse_categorical_crossentropy'
        },
        metrics={'class_output': metrics.SparseCategoricalAccuracy(name='accuracy')}
    )
    return model

if __name__ == "__main__":
    server_model = create_model()
    test_ds = load_test_dataset(config1.TEST_RECORD_PATH)

    # set up socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((config1.SERVER_IP, config1.PORT))
    server.listen(config1.CLIENTS_COUNT)
    print("📡 Server listening...")

    conns = []
    for i in range(config1.CLIENTS_COUNT):
        conn, addr = server.accept()
        conn.settimeout(10000)
        print(f"🔗 Client {i} connected from {addr}")
        conns.append(conn)

    for rnd in range(config1.NUM_ROUNDS):
        print(f"\n🔁 Round {rnd+1}")

        # Broadcast global weights
        wb  = pickle.dumps(server_model.get_weights())
        hdr = struct.pack(">I", len(wb))
        for c in conns:
            c.sendall(hdr + wb)

        collected   = []
        metrics_log = []

        # Receive updates from each client
        for idx, c in enumerate(conns):
            try:
                sz = struct.unpack(">I", c.recv(4))[0]
                data = b""
                while len(data) < sz:
                    pkt = c.recv(4096)
                    if not pkt: break
                    data += pkt
                payload = pickle.loads(data)
                collected.append(payload['weights'])

                joules   = payload.get('joules', 0.0)
                duration = payload.get('duration', 0.0)
                acc      = payload.get('accuracy', None)

                power = joules / duration if duration > 0 else 0.0
                metrics_log.append((idx, joules, duration, power, acc))
            except Exception as e:
                print(f"⚠️ Client {idx} failed: {e}")

        if collected:
            # Aggregate weights
            avg_w = [np.mean(ws, axis=0) for ws in zip(*collected)]
            server_model.set_weights(avg_w)
            print("✅ Aggregated weights applied.")

            # Log client metrics, handling missing accuracy
            for cid, joules, sec, power, acc in metrics_log:
                acc_str = f"{acc*100:.2f}%" if acc is not None else "N/A"
                print(f"   • Client {cid}: {joules:.1f} J, {sec:.1f}s → {power:.1f} W, acc={acc_str}")

            # Evaluate on test set
            res = server_model.evaluate(test_ds, verbose=0)
            print(
                f"📊 Eval → TotalLoss={res[0]:.4f}, "
                f"BBox={res[1]:.4f}, ClassLoss={res[2]:.4f}, Acc={res[3]*100:.2f}%"
            )
        else:
            print("⚠️ No updates received this round.")

    # Save final model and close connections
    server_model.save("federated_light_model.h5")
    print("📁 Model saved.")
    for c in conns:
        c.close()
