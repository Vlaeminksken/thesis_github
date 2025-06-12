# Federated Learning Pipeline (Thesis Project)

This repository contains the code and resources for a master's thesis project by Joran Verheijen (Joran.Verheijen@vub.be) and Robbe Vlaeminck (Robbe.Vlaeminck@vub.be). The project investigates the practical deployment of Federated Learning (FL) on the Jetson Nano. The primary objective is to evaluate how FL can deliver privacy-preserving, energy-efficient, and accurate object detection models on edge devices.

## Project Overview

Traditional centralized machine learning systems require users to send raw data to remote servers for training, introducing privacy risks and network bottlenecks. These concerns are especially important in assistive applications involving wearable cameras and personal sensing devices. FL addresses these issues by enabling decentralized training: user data remains on-device, and only model parameters (weights or gradients) are exchanged with a central aggregator. This approach enhances privacy, reduces bandwidth and energy consumption, and is well-suited for battery-powered edge devices.

This project develops a complete FL pipeline, including model design, client-server coordination, and systematic evaluation under varying configurations. The focus is on experimental evaluation and practical performance trade-offs in federated embedded learning.

## Research Objectives

- **Build a Functional FL Framework:** Develop a modular FL framework with a central aggregation server and multiple clients (Jetson Nano hardware and simulated nodes). Validate the framework using both the MNIST classification task and the Outdoor Obstacle Detection (OOD) dataset.
- **Decentralized Training for Privacy:** Ensure all training data remains on local devices, reducing privacy concerns and network traffic by exchanging only model parameters.
- **Compare FL with Centralized and Stand-Alone Learning:** Conduct controlled experiments to compare federated, centralized, and stand-alone learning. Evaluation metrics include classification accuracy, energy consumption, communication overhead, and convergence speed. Due to computational constraints, a custom lightweight CNN4 was developed for the OOD dataset instead of heavier architectures like YOLO or MobileNet.

## Key Contributions

- A modular and reproducible FL framework for embedded AI applications.
- Comparative analysis of federated, centralized, and stand-alone learning setups.
- A validated lightweight CNN model optimized for Jetson Nano.
- Design and deployment insights for FL in energy-constrained, privacy-sensitive environments.

## Project Structure

- `dataomvormen_yolo.py`: Data transformation scripts for YOLO.
- `Comparision/`: Contains scripts and models for comparing different YOLO and OOD approaches.
- `documents/`: Project documentation and Gantt chart scripts.
- `Improvements/object_detection1/`: Improved object detection scripts and federated learning models.
- `MNIST/`: Scripts and data for MNIST experiments.
- `Object_detect+utils/`: Centralized and federated learning scripts, configuration files, and utility functions.
- `Object_detection/`: Object detection scripts and configurations.
- `OOD.v1i.yolov5pytorch/`: OOD detection experiments with YOLOv5 in PyTorch.
- `standalone/`: Standalone scripts and models for YOLO and CNN experiments.

## Datasets
- **MNIST Dataset:** [https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)
- **Outdoor Obstacle Detection (OOD) Dataset:** [https://universe.roboflow.com/fpn/ood-pbnro](https://universe.roboflow.com/fpn/ood-pbnro)
- **Visually Impaired (VI) Dataset:** [https://universe.roboflow.com/all-mix/visually-impaired-dataset](https://universe.roboflow.com/all-mix/visually-impaired-dataset)

## Getting Started

1. Clone the repository:
   ```powershell
   git clone https://github.com/Vlaeminksken/thesis_github
   ```
2. Install required Python packages (see individual script requirements).
3. Run scripts as needed for your experiments.

## Requirements

- Python 3.7+
- TensorFlow, PyTorch, and other dependencies (see script headers or requirements files)
- Jetson Nano DNN image ([Qengineering/Jetson-Nano-image](https://github.com/Qengineering/Jetson-Nano-image) recommended)


## Practical Setup Notes

- **Jetson Nano DNN Image:** Instead of manually installing JetPack SDK, TensorFlow, and CUDA, use the Jetson Nano DNN image. This approach saves significant setup time and ensures compatibility, especially with CUDA since conflicts can/will happen.
- **microSD Card Size:** The default DNN image is for 32GB microSD cards. For this project, modify the image to work with a 64GB card to provide sufficient swap space for model training with GParted.
- **Flashing the Image:** Use the Raspberry Pi Foundation's Imager tool with the "no filtering" and "custom" options to flash the image to the microSD card. The process takes about 30 minutes.
- **WiFi Module Compatibility:** The TP-Link TL-WN722N USB WiFi module is recommended for reliable out-of-the-box support. Avoid the TP-Link Archer T2U. Note that the TL-WN722N may block other USB ports, so plan USB device connections accordingly.
- **Jetson Nano Hardware Limitations:** The Jetson Nano P3450 (2019) may only support inference (not training) for some algorithms (e.g., YOLO). This limitation motivated the use of the lightweight CNN4 model for training.
- **Python Package Versions:** The repository's Python scripts do not specify exact package versions. On Jetson Nano DNN, scripts generally run without additional updates (except possibly NumPy) as of June 2025. On Windows 11, use the latest package versions via pip.
- **Data Format:** The tf.record format is used for importing images from RoboFlow.
- **Configuration Files:** Configuration files (e.g., `configX.py`) are imported as modules (e.g., `import configX as cfg`) in the main scripts. This allows flexible configuration management for different experiments.

## Set up virtual environment
It is recommended to use a virtual environment to ensure packages are installed correctly. This ensures the portability of the code across different machines. The `uv` package manager works well for this purpose, and is also suggested by the Marimo project. 

See the [uv installation guide](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

```
uv venv                              # setup virtual environment (venv) with uv
.venv\Scripts\activate               # activate venv (Windows)
#source .venv/bin/activate                  # activate venv (Linux/OSX)
uv pip install -r requirements.txt   # install marimo environment & dependencies
```

## Usage

- Refer to the scripts in each folder for specific experiments and usage instructions.
- Configuration files are provided in the `Configs/` directories.

## Results

- Model weights, training logs, and evaluation plots are stored in the respective `models/` directories.

## Documentation

- See `Master-Project-Description.pdf` and other documents in the `documents/` folder for more details.
