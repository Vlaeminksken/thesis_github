import os
import tensorflow as tf
import re
from tqdm import tqdm

# === 1. Label map laden (uit pbtxt) ===
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
            id_to_name[id] = name
            name_to_id[name] = id
    return id_to_name, name_to_id

# === 2. TFRecord parsing ===
def parse_example(example_proto):
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    }
    return tf.io.parse_single_example(example_proto, features)

# === 3. Conversie uitvoeren per split ===
def convert_split(split, input_dir, output_dir, label_map):
    tfrecord_path = os.path.join(input_dir, split, 'Objects.tfrecord')
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_example)

    img_out_dir = os.path.join(output_dir, 'images', split)
    lbl_out_dir = os.path.join(output_dir, 'labels', split)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    for i, example in enumerate(tqdm(parsed_dataset, desc=f'Converting {split}')):
        image_data = example['image/encoded'].numpy()
        labels = tf.sparse.to_dense(example['image/object/class/label']).numpy()
        xmins = tf.sparse.to_dense(example['image/object/bbox/xmin']).numpy()
        xmaxs = tf.sparse.to_dense(example['image/object/bbox/xmax']).numpy()
        ymins = tf.sparse.to_dense(example['image/object/bbox/ymin']).numpy()
        ymaxs = tf.sparse.to_dense(example['image/object/bbox/ymax']).numpy()

        img_filename = f"{split}_{i:05d}.jpg"
        label_filename = f"{split}_{i:05d}.txt"

        img_path = os.path.join(img_out_dir, img_filename)
        with open(img_path, 'wb') as img_file:
            img_file.write(image_data)

        # YOLO labels: class_id x_center y_center width height (genormaliseerd)
        label_path = os.path.join(lbl_out_dir, label_filename)
        with open(label_path, 'w') as lbl_file:
            for cls, xmin, xmax, ymin, ymax in zip(labels, xmins, xmaxs, ymins, ymaxs):
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                lbl_file.write(f"{cls-1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# === 4. Main script ===
if __name__ == '__main__':
    input_dir = 'OOD.v1i.tfrecord'
    output_dir = 'OOD_YOLO'
    label_map_path = os.path.join(input_dir, 'train', 'Objects_label_map.pbtxt')

    id_to_name, name_to_id = load_label_map(label_map_path)
    print(f"LABELS ({len(id_to_name)}):", id_to_name)

    for split in ['train', 'valid', 'test']:
        convert_split(split, input_dir, output_dir, id_to_name)
