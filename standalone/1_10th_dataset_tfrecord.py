import os
import tensorflow as tf
import random
import shutil

def subset_tfrecord(input_path, output_path, fraction):
    raw_dataset = tf.data.TFRecordDataset(input_path)
    examples = list(raw_dataset)
    subset_size = max(1, int(len(examples) * fraction))
    subset_examples = random.sample(examples, subset_size)

    with tf.io.TFRecordWriter(output_path) as writer:
        for example in subset_examples:
            writer.write(example.numpy())

    print(f"✅ Geschreven: {subset_size} → {output_path}")

def main():
    base_dir = "C:/Users/robbe/Desktop/VUB/master/thesis_github"
    src = os.path.join(base_dir, "OOD.v1i.tfrecord")
    dst = os.path.join(base_dir, "standalone", "tfrecord")

    os.makedirs(os.path.join(dst, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst, "test_combined"), exist_ok=True)

    # ⬇️ Subset van train
    subset_tfrecord(
        input_path=os.path.join(src, "train", "Objects.tfrecord"),
        output_path=os.path.join(dst, "train", "Objects.tfrecord"),
        fraction=0.1
    )

    # ⬇️ Combineer valid + test → test_combined
    valid = list(tf.data.TFRecordDataset(os.path.join(src, "valid", "Objects.tfrecord")))
    test = list(tf.data.TFRecordDataset(os.path.join(src, "test", "Objects.tfrecord")))
    combined = valid + test
    subset_size = max(1, int(len(combined) * 0.1))
    subset_combined = random.sample(combined, subset_size)

    with tf.io.TFRecordWriter(os.path.join(dst, "test_combined", "Objects.tfrecord")) as writer:
        for example in subset_combined:
            writer.write(example.numpy())

    print(f"✅ Valid + Test samengevoegd → test_combined")

    # Kopieer de pbtxt van train naar beide
    label_src = os.path.join(src, "train", "Objects_label_map.pbtxt")
    for folder in ["train", "test_combined"]:
        label_dst = os.path.join(dst, folder, "Objects_label_map.pbtxt")
        shutil.copyfile(label_src, label_dst)

    print("✅ Label map gekopieerd naar beide mappen")

if __name__ == "__main__":
    main()
