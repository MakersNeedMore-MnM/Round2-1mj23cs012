"""
Drishti Kavach: Cross-Platform Local Jupyter Gateway Launcher for Google Colab
Compatible with macOS (Apple Silicon M-Series / Intel) and Windows 10 / 11 (NVIDIA CUDA / CPU).

Usage:
  # On Mac:
  .venv/bin/python src/4_colab_training/launch_colab_server.py

  # On Windows:
  python src/4_colab_training/launch_colab_server.py
"""

import os
import sys
import subprocess
import platform
import argparse


def launch_server(port: int = 8888):
    os_name = platform.system()
    machine = platform.machine()
    
    print("=" * 80)
    print(" 🚀 DRISHTI-KAVACH: LOCAL RUNTIME SERVER FOR GOOGLE COLAB")
    print("=" * 80)
    print(f" • Operating System : {os_name} ({machine})")
    print(f" • Python Runtime   : {sys.executable}")
    print(f" • Gateway Port     : {port}")
    print(" • Storage Mode     : Reads local 'dataset_rail-drishti/' directly (NO cloud upload needed)")
    print("=" * 80 + "\n")

    # Step 1: Install & enable jupyter_http_over_ws extension
    print("[1/2] Checking & enabling Google Colab WebSocket bridge...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", 
            "jupyter", "jupyter_http_over_ws"
        ])
        subprocess.check_call([
            sys.executable, "-m", "jupyter", "serverextension", "enable", "--py", "jupyter_http_over_ws"
        ])
        print("  ✓ Colab WebSocket extension is active and enabled.\n")
    except Exception as e:
        print(f"  [!] Note: WebSocket check completed: {e}\n")

    # Step 2: Launch Jupyter Notebook Server
    print("[2/2] Starting Jupyter Gateway for https://colab.research.google.com ...")
    print("-" * 80)
    print(" 📋 NEXT STEPS IN GOOGLE COLAB:")
    print(" 1. Copy the URL printed below containing 'http://localhost:8888/?token=...'")
    print(" 2. In Google Colab, click the [Connect dropdown] (top-right) -> 'Connect to a local runtime'")
    print(" 3. Paste the URL and click 'Connect'!")
    print("-" * 80 + "\n")

    cmd = [
        sys.executable, "-m", "jupyter", "notebook",
        "--NotebookApp.allow_origin=https://colab.research.google.com",
        f"--port={port}",
        "--NotebookApp.port_retries=0",
        "--no-browser"
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n[*] Local Colab Server stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch local Jupyter runtime for Google Colab")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind server on (default: 8888)")
    args = parser.parse_args()
    launch_server(port=args.port)
