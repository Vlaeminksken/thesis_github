import os

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths (using original filenames)
TRAIN_IMAGES_PATH = os.path.join(BASE_DIR, "task2", "train-images.idx3-ubyte")
TRAIN_LABELS_PATH = os.path.join(BASE_DIR, "task2", "train-labels.idx1-ubyte")
TEST_IMAGES_PATH = os.path.join(BASE_DIR, "task2", "t10k-images.idx3-ubyte")
TEST_LABELS_PATH = os.path.join(BASE_DIR, "task2", "t10k-labels.idx1-ubyte")

# Server settings
SERVER_IP = "172.20.10.4"
PORT = 5000

# Federated Learning settings
NUM_ROUNDS = 5
CLIENTS_COUNT = 2  # Change dynamically
LEARNING_RATE = 0.005
EPOCHS_PER_ROUND = 3
