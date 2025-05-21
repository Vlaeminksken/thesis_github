import os

BASE_DIR = r"C:\Users\joran\thesis_github"
# Corrected paths to TFRecord files OOD dataset
TRAIN_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord", "train", "Objects.tfrecord")
VALID_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord","valid", "Objects.tfrecord")
TEST_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord","test","Objects.tfrecord")
TRAIN_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_02", "train", "train.tfrecord")
VALID_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_02", "valid", "valid.tfrecord")
TEST_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_02", "test", "test.tfrecord")

# Optional: label map if needed
LABEL_MAP_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord", "train", "Objects_label_map.pbtxt")

IMAGE_HEIGHT    = 128
IMAGE_WIDTH     = 128
NUM_CLASSES     = 22     # ← your new dataset’s class count
NUM_BOXES       = 5      # max boxes per image
LEARNING_RATE   = 1e-4
