import os
import tensorflow as tf
import re
import subprocess
import time

# === 1. Load Label Map ===
def load_label_map(path):
    with open(path, 'r') as f:
        content = f.read()
    items = re.findall(r"item \{(.*?)\}", content, re.DOTALL)
    id_to_name = {}
    name_to_id = {}
    for item in items:
        id_match = re.search(r"id: (\d+)", item)
        name_match = re.search(r"name: \"(.*?)\"", item)
        if id_match and name_match:
            id = int(id_match.group(1))
            name = name_match.group(1)
            id_to_name[id - 1] = name  # zet naar zero-based
            name_to_id[name] = id - 1
    return id_to_name, name_to_id

# === 2. TFRecord Parsing ===
def parse_tfrecord_fn(example, name_to_id):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, [224, 224]) / 255.0

    labels = tf.sparse.to_dense(example['image/object/class/label'])
    label = tf.cast(labels[0] - 1, tf.int32)  # label naar zero-based
    return image, label

# === 3. Data Pipeline ===
def prepare_datasets(tfrecord_dir, name_to_id):
    def get_dataset(split):
        raw = tf.data.TFRecordDataset(os.path.join(tfrecord_dir, split, 'Objects.tfrecord'))
        parsed = raw.map(lambda x: parse_tfrecord_fn(x, name_to_id), num_parallel_calls=tf.data.AUTOTUNE)
        return parsed.shuffle(512).batch(32).prefetch(tf.data.AUTOTUNE)

    return get_dataset('train'), get_dataset('valid'), get_dataset('test')

# === 4. Model Bouwen ===
def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# === 5. Energie Logging via NVIDIA-SMI ===
def log_gpu_power_during(func):
    def wrapper(*args, **kwargs):
        power_log = []
        def log():
            while not done[0]:
                result = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])
                power = float(result.decode().strip().split('\n')[0])
                power_log.append(power)
                time.sleep(0.2)

        done = [False]
        import threading
        t = threading.Thread(target=log)
        t.start()
        result = func(*args, **kwargs)
        done[0] = True
        t.join()

        avg_power = sum(power_log) / len(power_log)
        print(f"Gemiddeld GPU-verbruik tijdens test: {avg_power:.2f} Watt")
        return result
    return wrapper

# === 6. Training en Evaluatie ===
if __name__ == '__main__':
    root_path = r'C:/Users/robbe/Desktop/VUB/master/thesis_github/standalone/tfrecord_dataset'
    label_path = os.path.join(root_path, 'train', 'Objects_label_map.pbtxt')
    model_output_path = r'C:/Users/robbe/Desktop/VUB/master/thesis_github/standalone/models/cnn_model.h5'

    id_to_name, name_to_id = load_label_map(label_path)
    num_classes = len(id_to_name)

    def get_dataset(split):
        raw = tf.data.TFRecordDataset(os.path.join(root_path, split, 'Objects.tfrecord'))
        parsed = raw.map(lambda x: parse_tfrecord_fn(x, name_to_id), num_parallel_calls=tf.data.AUTOTUNE)
        return parsed.shuffle(512).batch(32).prefetch(tf.data.AUTOTUNE)

    train_ds = get_dataset('train')
    test_ds = get_dataset('test_combined')

    model = build_model(num_classes)

    model.fit(train_ds, epochs=10)  # zonder validatie

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)
    print(f"Model opgeslagen in: {model_output_path}")

    @log_gpu_power_during
    def evaluate():
        loss, acc = model.evaluate(test_ds)
        print(f"Test accuraatheid: {acc:.4f}")

    evaluate()

