import os

BASE_DIR = r"C:\Users\joran\thesis_github"
# Corrected paths to TFRecord files
"""TRAIN_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord", "train", "Objects.tfrecord")
VALID_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord","valid", "Objects.tfrecord")
TEST_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord","test","Objects.tfrecord")"""

TRAIN_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_01", "train", "train.tfrecord")
VALID_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_01", "valid", "valid.tfrecord")
TEST_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_01", "test", "test.tfrecord")
# Optional: label map if needed
#LABEL_MAP_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord", "train", "Objects_label_map.pbtxt")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "Visually impaired dataset.v2i.tfrecord", "train", "multipleobjects_label_map.pbtxt")

# Server/Federated learning config
SERVER_IP = "10.12.10.84"
PORT = 5000
NUM_ROUNDS = 40
CLIENTS_COUNT = 10
LEARNING_RATE = 0.005
EPOCHS_PER_ROUND = 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_CLASSES = 30
NUM_BOXES = 5
CPU_TDP = 55

