#!/usr/bin/env python3
"""
launch_clients.py ───────────────────────────────────────────────────────────
Start N federated-learning clients as separate OS processes.

• Works on Windows, Linux, macOS.
• Each client gets its own --client_id 0…N-1.
• Logs from every client go to   logs/client_<id>.txt
  (so your terminal stays clean).

Usage
─────
python launch_clients.py 50               # start 50 clients
python launch_clients.py 50 --delay 2     # 2-second gap between launches
"""
import argparse, os, sys, time, subprocess, pathlib, platform

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("num_clients", type=int, help="How many client processes?")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="Seconds to wait between launches (default 0)")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use (default = this one)")
    args = ap.parse_args()

    logs_dir = pathlib.Path("logs");  logs_dir.mkdir(exist_ok=True)

    procs = []
    for cid in range(args.num_clients):
        log_file = logs_dir / f"client_{cid:02d}.txt"
        cmd = [args.python, "client.py", "--client_id", str(cid)]

        # On Windows open each client in a *new* console window so you can peek.
        # Comment out the creationflags line if you prefer everything headless.
        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_CONSOLE

        p = subprocess.Popen(
            cmd,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            creationflags=creationflags
        )
        procs.append(p)
        print(f"🚀 Launched client {cid:02d}  (pid {p.pid}) → {log_file}")
        time.sleep(args.delay)

    print(f"✅ {len(procs)} clients launched. They will keep running until "
          f"they finish or you terminate them (Ctrl-C here stops nothing).")

if __name__ == "__main__":
    main()