import os
import time
import psutil
import sys
import torch
import subprocess
import threading

# Pad naar yolov5-repo
YOLO_PATH = os.path.abspath("yolov5")
sys.path.insert(0, YOLO_PATH)

# ---- PATCH START ----
# Monkey patch de confusion matrix class om fouten te voorkomen
from utils import metrics

original_process_batch = metrics.ConfusionMatrix.process_batch

def patched_process_batch(self, detections, labels):
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.size(1) < 1:
        return
    mask = (labels[:, 0] < self.nc)
    labels = labels[mask]
    if detections is not None and len(detections):
        detections = detections[detections[:, 5] < self.nc]
    return original_process_batch(self, detections, labels)

metrics.ConfusionMatrix.process_batch = patched_process_batch
# ---- PATCH END ----

from val import run as val_run  # na patch importeren!

def monitor_gpu_power(samples, interval=0.1, running_flag=None):
    """Meet GPU power (in Watt) via nvidia-smi."""
    while running_flag["running"]:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            power = float(result.stdout.strip().split('\n')[0])
            samples.append(power)
            time.sleep(interval)
        except Exception:
            break

def measure_power_usage(func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    cpu_samples, gpu_samples = [], []

    # Start GPU monitoring thread met controle-vlag
    flag = {"running": True}
    gpu_thread = threading.Thread(target=monitor_gpu_power, args=(gpu_samples, 0.1, flag))
    gpu_thread.start()

    # Start evaluatie
    start = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start

    # Stop de GPU thread
    flag["running"] = False
    gpu_thread.join(timeout=1)

    # Meet CPU na afloop
    for _ in range(int(duration * 10)):
        cpu_samples.append(process.cpu_percent(interval=0.1))

    cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    gpu_avg = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0
    return result, cpu_avg, gpu_avg


if __name__ == "__main__":
    data_yaml_path = os.path.abspath(r"C:\Users\robbe\Desktop\VUB\master\thesis_github\standalone\yolo_dataset\data_standalone.yaml")
    weights_path = os.path.abspath(r"C:\Users\robbe\Desktop\VUB\master\thesis_github\yolov5\yolov5m.pt")

    print("🚀 Evaluatie met COCO-voorgetraind model (yolov5m.pt) gestart...\n")

    results, cpu_power, gpu_power = measure_power_usage(
        val_run,
        weights=weights_path,
        data=data_yaml_path,
        task='test',
        save_json=False,
        save_txt=False,
        verbose=False
    )

    mAP_50 = results[2][0]
    #mAP_50_95 = results[3][0]

    print(f"\n✅ Gemiddelde accuraatheid over ALLE 22 klassen (mAP@0.5): {mAP_50*100:.2f}%")
    #print(f"📊 mAP@[.5:.95]: {mAP_50_95*100:.2f}%")
    print(f"⚡ Gemiddeld CPU energieverbruik: ~{cpu_power:.2f}% CPU-gebruik")
    print(f"⚡ Gemiddeld GPU energieverbruik: ~{gpu_power:.2f} Watt via NVIDIA-SMI")
