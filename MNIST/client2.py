import socket
import pickle
import struct
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import config  # Import global settings
import sys  # For command-line client ID

# Get client ID (Each client must pass an ID as an argument)
if len(sys.argv) != 2:
    print("Usage: python client.py <client_id>")
    sys.exit(1)

CLIENT_ID = int(sys.argv[1])  # Example: 0 for first client, 1 for second, etc.

def load_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)
    return images / 255.0  # Normalize

def load_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return keras.utils.to_categorical(labels, 10)  # One-hot encode

# Load dataset
train_images = load_idx_images(config.TRAIN_IMAGES_PATH)
train_labels = load_idx_labels(config.TRAIN_LABELS_PATH)

# **Dynamically Assign Dataset Split**
num_clients = config.CLIENTS_COUNT
total_samples = len(train_images)

samples_per_client = total_samples // num_clients  # Equal split

start_idx = CLIENT_ID * samples_per_client
end_idx = (CLIENT_ID + 1) * samples_per_client if CLIENT_ID != num_clients - 1 else total_samples

client_images = train_images[start_idx:end_idx]
client_labels = train_labels[start_idx:end_idx]

print(f"Client {CLIENT_ID}: Using dataset indices {start_idx} to {end_idx} ({len(client_images)} samples)")

def create_model():
    model = keras.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

client_model = create_model()

# Connect to server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((config.SERVER_IP, config.PORT))
print(f"Client {CLIENT_ID} connected to server")

for round in range(config.NUM_ROUNDS):
    print(f"Training Round {round+1}/{config.NUM_ROUNDS}")

    # Receive model weights
    data_length = struct.unpack(">I", client.recv(4))[0]
    received_data = b""
    while len(received_data) < data_length:
        received_data += client.recv(4096)

    client_model.set_weights(pickle.loads(received_data))

    # Train model
    client_model.fit(client_images, client_labels, epochs=config.EPOCHS_PER_ROUND, verbose=1)

    # Send updated weights
    serialized_data = pickle.dumps(client_model.get_weights())
    client.sendall(struct.pack(">I", len(serialized_data)) + serialized_data)

client.close()
print(f"Client {CLIENT_ID} finished training and disconnected")
