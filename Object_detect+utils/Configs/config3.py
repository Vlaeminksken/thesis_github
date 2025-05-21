import os

BASE_DIR = r"C:\Users\joran\thesis_github"
# Corrected paths to TFRecord files
TRAIN_RECORD_PATH = os.path.join(BASE_DIR, "Visually impaired dataset.v2i.tfrecord", "train", "multipleobjects.tfrecord")
VALID_RECORD_PATH = os.path.join(BASE_DIR, "Visually impaired dataset.v2i.tfrecord","valid", "multipleobjects.tfrecord")
TEST_RECORD_PATH = os.path.join(BASE_DIR, "Visually impaired dataset.v2i.tfrecord","test","multipleobjects.tfrecord")

# Optional: label map if needed
LABEL_MAP_PATH = os.path.join(BASE_DIR, "Visually impaired dataset.v2i.tfrecord", "train", "multipleobjects_label_map.pbtxt")

# Server/Federated learning config
SERVER_IP = "10.9.64.169"
PORT = 5000
NUM_ROUNDS = 20
CLIENTS_COUNT = 1
LEARNING_RATE = 0.005
EPOCHS_PER_ROUND = 1
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_CLASSES = 30
NUM_BOXES = 5
CPU_TDP = 55

