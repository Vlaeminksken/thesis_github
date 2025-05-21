import os

BASE_DIR = r"C:\Users\joran\thesis_github"
# Corrected paths to TFRecord files
"""TRAIN_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_44", "train", "train.tfrecord")
VALID_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_44", "valid", "valid.tfrecord")
TEST_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits","split_44", "test", "test.tfrecord")
"""
TRAIN_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord", "train", "Objects.tfrecord")
VALID_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord","valid", "Objects.tfrecord")
TEST_RECORD_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord","test","Objects.tfrecord")
"""
TRAIN_RECORD_PATH = os.path.join(BASE_DIR, "Visually impaired dataset.v2i.tfrecord", "train", "multipleobjects.tfrecord")
VALID_RECORD_PATH = os.path.join(BASE_DIR, "Visually impaired dataset.v2i.tfrecord","valid", "multipleobjects.tfrecord")
TEST_RECORD_PATH = os.path.join(BASE_DIR, "Visually impaired dataset.v2i.tfrecord","test","multipleobjects.tfrecord")"""
# Optional: label map if needed
# LABEL_MAP_PATH = os.path.join(BASE_DIR, "OOD.v1i.tfrecord", "train", "Objects_label_map.pbtxt")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "Visually impaired dataset.v2i.tfrecord", "train", "multipleobjects_label_map.pbtxt")
#SPLITS_DIR = os.path.join(BASE_DIR, "OOD.v1i.tfrecord_splits")
SPLITS_DIR = os.path.join(BASE_DIR, "Visually impaired dataset.v2i.tfrecord_splits1")
# Server/Federated learning config
SERVER_IP = "172.20.10.4"
PORT = 5000
NUM_ROUNDS = 20
CLIENTS_COUNT = 10
LEARNING_RATE = 0.005
EPOCHS_PER_ROUND = 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_CLASSES = 3
NUM_BOXES = 1
CPU_TDP = 55
BATCH_SIZE = 8
MOMENTUM =0.9
SERVER_LR = 1.0
PROX_MU = 0.01
CLIP_NORM =1
MODE = "transfer"
PRETRAINED_PATH = r"C:\Users\joran\thesis_github\fl_classification_3class1.keras"
NUM_FINE_CLASSES = 30           # or 30, etc.
MOVING_IDS       = [0, 3, 6, 7, 8, 9, 16, 18, 27] #[1, 2, 4, 7, 9, 10, 17, 19]   # zero-based IDs that can move [0, 3, 6, 7, 8, 9, 16, 18, 27]
NUM_CLASSES      = 3            # leave this at 3
