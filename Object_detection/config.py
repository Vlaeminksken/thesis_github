import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Corrected paths to TFRecord files
TRAIN_RECORD_PATH = os.path.join(BASE_DIR, "dataset", "train", "Objects_half.tfrecord")
VALID_RECORD_PATH = os.path.join(BASE_DIR, "dataset", "valid", "Objects.tfrecord")
TEST_RECORD_PATH = os.path.join(BASE_DIR, "dataset", "test", "Objects.tfrecord")

# Optional: label map if needed
LABEL_MAP_PATH = os.path.join(BASE_DIR, "dataset", "train", "Objects_label_map.pbtxt")

# Server/Federated learning config
SERVER_IP = "172.20.10.4"
PORT = 5000
NUM_ROUNDS = 3
CLIENTS_COUNT = 2
LEARNING_RATE = 0.005
EPOCHS_PER_ROUND = 3
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_CLASSES = 22
NUM_BOXES = 5
