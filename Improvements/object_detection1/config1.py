import os

BASE_DIR = r"C:\Users\joran\thesis_github"
# Corrected paths to TFRecord files
TRAIN_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_02", "train", "train.tfrecord")
VALID_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_02", "valid", "valid.tfrecord")
TEST_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_02", "test", "test.tfrecord")

# Optional: label map if needed
LABEL_MAP_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord", "train", "Objects_label_map.pbtxt")

# Server/Federated learning config
SERVER_IP = "172.20.10.4"
PORT = 6000
NUM_ROUNDS = 5
CLIENTS_COUNT = 10
LEARNING_RATE = 0.005
EPOCHS_PER_ROUND = 5
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_CLASSES = 22
NUM_BOXES = 5
CPU_TDP = 55