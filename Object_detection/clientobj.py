# === CLIENT ===

import socket, pickle, struct, os, time, tensorflow as tf
from tensorflow.keras import layers, models
import config

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # optional: run on CPU to save memory

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

def load_dataset(record_path, client_id, total_clients):
    ds = tf.data.TFRecordDataset(record_path)
    ds = ds.shuffle(100).map(parse_tfrecord).enumerate()
    ds = ds.filter(lambda i, d: tf.equal(i % total_clients, client_id))
    ds = ds.take(500)
    ds = ds.map(lambda i, d: d).padded_batch(1).prefetch(1)
    return ds

def create_model():
    inputs = tf.keras.Input(shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    bbox = layers.Dense(config.NUM_BOXES * 4, activation='sigmoid')(x)
    bbox = layers.Reshape((config.NUM_BOXES, 4), name='bbox_output')(bbox)
    cls = layers.Dense(config.NUM_BOXES * config.NUM_CLASSES, activation='softmax')(x)
    cls = layers.Reshape((config.NUM_BOXES, config.NUM_CLASSES), name='class_output')(cls)
    model = models.Model(inputs, [bbox, cls])
    model.compile(optimizer='adam',
                  loss={'bbox_output': 'mse', 'class_output': 'sparse_categorical_crossentropy'})
    return model

CLIENT_ID = 0
dataset = load_dataset(config.TRAIN_RECORD_PATH, CLIENT_ID, config.CLIENTS_COUNT)
model = create_model()

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((config.SERVER_IP, config.PORT))
print("✅ Connected to server.")

for rnd in range(config.NUM_ROUNDS):
    print(f"\n🔁 Round {rnd+1}")
    size = struct.unpack(">I", client.recv(4))[0]
    data = b""
    while len(data) < size:
        data += client.recv(4096)
    model.set_weights(pickle.loads(data))
    print("🧠 Training...")
    model.fit(dataset, epochs=config.EPOCHS_PER_ROUND, verbose=1)
    weights = pickle.dumps(model.get_weights())
    client.sendall(struct.pack(">I", len(weights)) + weights)

client.close()
print("🚪 Disconnected.")
