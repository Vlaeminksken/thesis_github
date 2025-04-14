# === CLIENT ===

import socket
import pickle
import struct
import numpy as np
import os
import time
import psutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

# Resource tuning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Load CLIENT_ID
try:
    CLIENT_ID = int(1)
except ValueError:
    CLIENT_ID = 0
print(f"🚀 Client {CLIENT_ID} is starting...")

# Load dataset
def load_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)
    return images

def load_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

train_images = load_idx_images(config.TRAIN_IMAGES_PATH) / 255.0
train_labels = keras.utils.to_categorical(load_idx_labels(config.TRAIN_LABELS_PATH), 10)

# Partition data
num_clients = config.CLIENTS_COUNT
total_samples = len(train_images)
samples_per_client = total_samples // num_clients
start_idx = CLIENT_ID * samples_per_client
end_idx = (CLIENT_ID + 1) * samples_per_client if CLIENT_ID != num_clients - 1 else total_samples

client_images = train_images[start_idx:end_idx][:300]
client_labels = train_labels[start_idx:end_idx][:300]
print(f"📦 Client {CLIENT_ID}: Using {len(client_images)} samples")

# Model
def create_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

client_model = create_model()

# Connect to server
MAX_ATTEMPTS = 10
for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(100)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        client.connect((config.SERVER_IP, config.PORT))
        print(f"✅ Client {CLIENT_ID} connected to server on attempt {attempt}")
        break
    except (ConnectionRefusedError, socket.timeout):
        print(f"❌ Client {CLIENT_ID} failed to connect (Attempt {attempt}/{MAX_ATTEMPTS})")
        time.sleep(5)
else:
    print(f"❌ Client {CLIENT_ID} could not connect after {MAX_ATTEMPTS} attempts. Exiting.")
    exit(1)

# Federated loop
for round in range(config.NUM_ROUNDS):
    print(f"\n🔄 Round {round+1}/{config.NUM_ROUNDS} started")
    try:
        data_length_bytes = client.recv(4)
        if not data_length_bytes:
            print("❌ Server closed connection.")
            break

        data_length = struct.unpack(">I", data_length_bytes)[0]
        print(f"⬇️ Receiving {data_length} bytes of weights...")

        received_data = b""
        while len(received_data) < data_length:
            packet = client.recv(min(4096, data_length - len(received_data)))
            if not packet:
                raise ConnectionResetError("Server disconnected")
            received_data += packet
            time.sleep(0.01)

        print(f"✅ Received {len(received_data)} bytes")
        weights = pickle.loads(received_data)
        client_model.set_weights(weights)

        print("✅ Ready to train. Model summary:")
        client_model.summary()

        print(f"🧠 Training on {len(client_images)} samples...")
        print(f"🔍 Memory Before Training: {psutil.virtual_memory().percent}%")
        start_time = time.time()

        client_model.fit(client_images, client_labels,
                         epochs=config.EPOCHS_PER_ROUND,
                         batch_size=4,
                         verbose=1)

        print(f"✅ Training complete in {time.time() - start_time:.2f} sec")
        print(f"🔍 Memory After Training: {psutil.virtual_memory().percent}%")

        # Send updated weights
        time.sleep(1)
        updated_weights = pickle.dumps(client_model.get_weights())
        print(f"⬆️ Sending {len(updated_weights)} bytes back to server...")
        client.sendall(struct.pack(">I", len(updated_weights)) + updated_weights)

    except (socket.timeout, ConnectionResetError) as e:
        print(f"❌ Connection error: {e}")
        break
    except Exception as ex:
        print(f"❌ Unexpected error: {ex}")
        break

client.close()
print(f"🔚 Client {CLIENT_ID} finished and disconnected.")
