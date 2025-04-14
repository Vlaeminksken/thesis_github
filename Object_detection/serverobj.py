# === SERVER ===

import socket, pickle, struct, time, os, psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import config

def parse_tfrecord(example_proto):
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }

    example = tf.io.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)) / 255.0

    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    labels = tf.sparse.to_dense(example['image/object/class/label']) - 1

    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
    boxes = boxes[:config.NUM_BOXES]
    labels = labels[:config.NUM_BOXES]

    boxes = tf.pad(boxes, [[0, config.NUM_BOXES - tf.shape(boxes)[0]], [0, 0]])
    labels = tf.pad(labels, [[0, config.NUM_BOXES - tf.shape(labels)[0]]])

    return image, (boxes, labels)

def load_test_dataset(path):
    return tf.data.TFRecordDataset(path).map(parse_tfrecord).padded_batch(1).prefetch(1)

test_dataset = load_test_dataset(config.TEST_RECORD_PATH)

def create_model():
    inputs = layers.Input(shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    bbox = layers.Dense(config.NUM_BOXES * 4, activation='sigmoid', name='bbox')(x)
    bbox = layers.Reshape((config.NUM_BOXES, 4), name='bbox_output')(bbox)

    cls = layers.Dense(config.NUM_BOXES * config.NUM_CLASSES, activation='softmax', name='class')(x)
    cls = layers.Reshape((config.NUM_BOXES, config.NUM_CLASSES), name='class_output')(cls)

    model = models.Model(inputs, [bbox, cls])
    model.compile(
        optimizer='adam',
        loss={
            'bbox_output': 'mse',
            'class_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'class_output': tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        }
    )
    return model

# === Federated Server Logic ===
server_model = create_model()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((config.SERVER_IP, config.PORT))
server.listen(config.CLIENTS_COUNT)
print("📡 Server listening...")

connections = []
while len(connections) < config.CLIENTS_COUNT:
    conn, addr = server.accept()
    conn.settimeout(10000)
    print(f"🔗 Client {len(connections)} connected from {addr}")
    connections.append(conn)

for rnd in range(config.NUM_ROUNDS):
    print(f"\n🔁 Round {rnd+1}")
    weights_bytes = pickle.dumps(server_model.get_weights())
    for conn in connections:
        conn.sendall(struct.pack(">I", len(weights_bytes)) + weights_bytes)

    new_weights = []
    for conn in connections:
        try:
            size = struct.unpack(">I", conn.recv(4))[0]
            data = b""
            while len(data) < size:
                packet = conn.recv(4096)
                if not packet: break
                data += packet
            client_weights = pickle.loads(data)
            new_weights.append(client_weights)
        except:
            continue

    if new_weights:
        avg_weights = [np.mean(w, axis=0) for w in zip(*new_weights)]
        server_model.set_weights(avg_weights)
        print("✅ Aggregated weights applied.")

        # Evaluate
        results = server_model.evaluate(test_dataset, verbose=0)
        print(f"📊 Eval -> Total: {results[0]:.4f}, BBox: {results[1]:.4f}, Class: {results[2]:.4f}, Accuracy: {results[3] * 100:.2f}%")
    else:
        print("⚠️ No updates received.")

server_model.save("federated_light_model.h5")
print("📁 Model saved.")
for conn in connections:
    conn.close()
