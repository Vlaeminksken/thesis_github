# === SERVER ===

import socket
import pickle
import struct
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import config
import time

# Load test data
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

test_images = load_idx_images(config.TEST_IMAGES_PATH) / 255.0
test_labels = keras.utils.to_categorical(load_idx_labels(config.TEST_LABELS_PATH), 10)

# Define model
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

# Setup server socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((config.SERVER_IP, config.PORT))
server.listen(config.CLIENTS_COUNT)
print(f"🖥️ Server listening on {config.SERVER_IP}:{config.PORT}")
print(f"⌛ Waiting for {config.CLIENTS_COUNT} clients to connect...")

connections = []
addresses = []
start_time = time.time()
MAX_WAIT_SECONDS = 600
server.settimeout(100)

while len(connections) < config.CLIENTS_COUNT:
    try:
        conn, addr = server.accept()
        conn.settimeout(100)
        connections.append(conn)
        addresses.append(addr)
        print(f"🔗 Client {len(connections)-1}/{config.CLIENTS_COUNT} connected from {addr}")
    except socket.timeout:
        elapsed = time.time() - start_time
        print(f"⌛ Waiting... {len(connections)}/{config.CLIENTS_COUNT} clients connected after {int(elapsed)}s")
        if elapsed > MAX_WAIT_SECONDS:
            print("⚠️ Timeout: Not all clients connected. Continuing with available clients.")
            break

server_model = create_model()

# Federated training
for round in range(config.NUM_ROUNDS):
    print(f"\n=== 🧐 Federated Round {round+1}/{config.NUM_ROUNDS} ===")
    weights_list = []

    model_weights = pickle.dumps(server_model.get_weights())
    for idx, conn in enumerate(connections):
        try:
            print(f"📤 Sending weights to Client {idx} ({len(model_weights)} bytes)")
            conn.sendall(struct.pack(">I", len(model_weights)) + model_weights)
        except (BrokenPipeError, socket.timeout):
            print(f"❌ Client {idx} connection error during send.")

    for idx, conn in enumerate(connections):
        try:
            data_length_bytes = conn.recv(4)
            if not data_length_bytes:
                print(f"⚠️ Client {idx} closed connection.")
                continue

            data_length = struct.unpack(">I", data_length_bytes)[0]
            print(f"📅 Receiving {data_length} bytes from Client {idx}")

            received_data = b""
            while len(received_data) < data_length:
                packet = conn.recv(min(4096, data_length - len(received_data)))
                if not packet:
                    print(f"⚠️ Client {idx} disconnected mid-transfer.")
                    break
                received_data += packet
                time.sleep(0.01)

            weights = pickle.loads(received_data)
            weights_list.append(weights)
            print(f"✅ Received weights from Client {idx}")

        except socket.timeout:
            print(f"⏳ Timeout receiving from Client {idx}")
        except ConnectionResetError:
            print(f"❌ Client {idx} disconnected unexpectedly.")
        except Exception as e:
            print(f"❌ Error receiving from Client {idx}: {e}")

    if weights_list:
        new_weights = [np.mean(w, axis=0) for w in zip(*weights_list)]
        server_model.set_weights(new_weights)
        print(f"✅ Aggregated model updated from {len(weights_list)} clients")
    else:
        print("⚠️ No client weights received this round.")

    loss, acc = server_model.evaluate(test_images, test_labels, verbose=0)
    print(f"📊 Validation Accuracy: {acc:.4f}")

server_model.save("federated_mnist_cnn.h5")
print("📂 Model saved as 'federated_mnist_cnn.h5'")

for conn in connections:
    try:
        conn.close()
    except:
        pass
print("🧹 Server shut down.")
