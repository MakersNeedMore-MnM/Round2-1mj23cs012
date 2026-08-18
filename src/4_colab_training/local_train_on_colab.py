"""
Drishti Kavach: Local Hardware Runtime for Google Colab (Mac & Windows Guide + Server Launcher)

========================================================================================
💡 HOW IT WORKS (Google Colab Local Runtime Architecture)
========================================================================================
When you connect Google Colab to a "Local Runtime":
  1. Google Colab in your web browser acts ONLY as the visual frontend / notebook interface.
  2. The code and training execute 100% LOCALLY on your computer's native hardware:
       - On macOS   : Runs on Apple Silicon M4 GPU via Metal Performance Shaders (MPS).
       - On Windows : Runs on your NVIDIA GPU (via CUDA) or multi-core CPU.
  3. 🚀 ZERO CLOUD UPLOAD NEEDED:
       Because execution happens on your machine, your code reads your local
       'dataset_rail-drishti/' directory directly at blazing-fast NVMe/PCIe SSD speeds!
       You DO NOT need to upload 'raildrishti_colab.zip' or anything to Google Drive.
  4. Best model checkpoints ('models/RailDrishti.pt') are saved DIRECTLY onto your computer's disk.

========================================================================================
🛠️ SETUP INSTRUCTIONS (STEP-BY-STEP)
========================================================================================

--- [OPTION A] macOS (Apple Silicon M4 / M-Series) ---
1. Open Terminal and navigate to the project directory:
   cd /Users/alvinsonny/Desktop/drishti-kavach

2. Ensure your virtual environment is active and install the Colab WebSocket bridge:
   .venv/bin/pip install jupyter jupyter_http_over_ws
   .venv/bin/jupyter serverextension enable --py jupyter_http_over_ws

3. Start the Jupyter server allowing Google Colab connections:
   .venv/bin/jupyter notebook \\
     --NotebookApp.allow_origin='https://colab.research.google.com' \\
     --port=8888 \\
     --NotebookApp.port_retries=0

4. Copy the localhost URL with token printed in your terminal (e.g. http://localhost:8888/?token=abcdef123456...).

5. In Google Colab:
   • Click the downward arrow next to the 'Connect' button in the top right corner.
   • Select "Connect to a local runtime".
   • Paste the copied URL into the backend URL field and click "Connect".

--- [OPTION B] Windows (NVIDIA CUDA / CPU) ---
1. Open PowerShell or Command Prompt in your repository folder:
   cd C:\\path\\to\\drishti-kavach

2. Install the WebSocket extension:
   pip install jupyter jupyter_http_over_ws
   jupyter serverextension enable --py jupyter_http_over_ws

3. Start the Jupyter server:
   jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0

4. Copy the localhost URL with token and paste it into Google Colab -> "Connect to a local runtime".

========================================================================================
🚀 ONE-CLICK AUTOMATIC LAUNCHER
========================================================================================
You can also launch the local server automatically using this script:
  python src/4_colab_training/local_train_on_colab.py --start
"""

import os
import sys
import subprocess
import argparse
import platform


def print_full_guide():
    """Prints the comprehensive setup guide in the terminal."""
    print(__doc__)


def start_local_colab_server(port: int = 8888):
    """
    Automates installation of jupyter_http_over_ws and launches the Jupyter server
    configured specifically for Google Colab web connections.
    """
    print("=" * 80)
    print(" 🚀 LAUNCHING LOCAL JUPYTER RUNTIME FOR GOOGLE COLAB")
    print("=" * 80)
    print(f" • Platform : {platform.system()} ({platform.machine()})")
    print(f" • Python   : {sys.executable}")
    print(f" • Port     : {port}")
    print(" • Note     : NO Google Drive upload needed. Reads local dataset directly!")
    print("=" * 80 + "\n")

    # 1. Install & enable jupyter_http_over_ws extension
    print("[1/2] Verifying 'jupyter_http_over_ws' extension...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jupyter", "jupyter_http_over_ws"])
        subprocess.check_call([sys.executable, "-m", "jupyter", "serverextension", "enable", "--py", "jupyter_http_over_ws"])
        print("  ✓ Extension verified and enabled successfully.\n")
    except Exception as e:
        print(f"[!] Warning: Could not enable extension automatically: {e}")

    # 2. Launch Jupyter Notebook Server
    print("[2/2] Starting Jupyter server allowing https://colab.research.google.com origin...")
    print("--------------------------------------------------------------------------------")
    print(" Instructions:")
    print(" 1. Look for the URL below with 'http://localhost:8888/?token=...'")
    print(" 2. In Google Colab, click [Connect dropdown] (top right) -> 'Connect to a local runtime'")
    print(" 3. Paste the URL and click 'Connect'!")
    print("--------------------------------------------------------------------------------\n")

    cmd = [
        sys.executable, "-m", "jupyter", "notebook",
        f"--NotebookApp.allow_origin=https://colab.research.google.com",
        f"--port={port}",
        "--NotebookApp.port_retries=0",
        "--no-browser"
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n[*] Local Colab Jupyter Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Hardware Runtime Guide & Server Launcher for Google Colab")
    parser.add_argument("--start", action="store_true", help="Automatically launch the local Jupyter server for Colab")
    parser.add_argument("--port", type=int, default=8888, help="Port to run Jupyter server on (default 8888)")
    args = parser.parse_args()

    if args.start:
        start_local_colab_server(port=args.port)
    else:
        print_full_guide()
        print("\n👉 To start the local server automatically, run:")
        print("   python src/4_colab_training/local_train_on_colab.py --start\n")
