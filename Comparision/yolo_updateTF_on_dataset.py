import subprocess
import time
import threading
import os

def log_gpu_power(log_file, stop_signal):
    with open(log_file, "w") as f:
        while not stop_signal[0]:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits']
            )
            power = float(result.decode().strip().split('\n')[0])
            f.write(f"{power}\n")
            time.sleep(0.2)

def run_yolo_val():
    current_script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(current_script_path))  # van Comparision naar thesis_github
    yolov5_path = os.path.join(base_dir, "yolov5")

    result = subprocess.run([
        "python", "val.py",
        "--weights", "C:/Users/robbe/Desktop/VUB/master/thesis_github/Comparision/models/yolov5m_ood/weights/best.pt",
        "--data", "OOD.v1i.yolov5pytorch\data.yaml",
        "--task", "test",
        "--img", "640"
    ], cwd=yolov5_path, capture_output=True, text=True)

    # Debug: altijd ook stderr dumpen
    full_output = result.stdout + "\n--- STDERR ---\n" + result.stderr
    return full_output


def parse_results(val_output, result_file):
    with open(result_file, "w") as f:
        f.write(val_output)

    print("✅ Accuraatheid gelogd in:", result_file)

def calculate_avg_power(log_file, summary_file):
    with open(log_file, "r") as f:
        power_values = [float(line.strip()) for line in f.readlines()]
    avg_power = sum(power_values) / len(power_values) if power_values else 0.0

    with open(summary_file, "w") as f:
        f.write(f"Gemiddeld energieverbruik (GPU): {avg_power:.2f} Watt\n")

    print("🔋 Energieverbruik opgeslagen in:", summary_file)

if __name__ == "__main__":
    power_log = "power_yolov5_raw.txt"
    result_log = "results_yolov5.txt"
    summary_log = "power_yolov5.txt"

    stop_signal = [False]
    t = threading.Thread(target=log_gpu_power, args=(power_log, stop_signal))
    t.start()

    print("🚀 Evaluatie gestart...")
    result = run_yolo_val()

    stop_signal[0] = True
    t.join()

    parse_results(result, result_log)
    calculate_avg_power(power_log, summary_log)
