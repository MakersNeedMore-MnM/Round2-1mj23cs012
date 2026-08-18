# 🌐 Drishti Kavach: Google Colab Local Runtime Training Guide (macOS & Windows)

This guide shows you how to train the **RailDrishti** model using the **Google Colab Notebook UI in your web browser**, while executing all computations, PyTorch neural network training, and dataset access directly on **your own laptop hardware (Apple Silicon MPS / Windows NVIDIA GPU / CPU)**.

---

## 🎯 Why Use Colab with a Local Runtime?
1. **Interactive Colab Interface**: Beautiful cell-by-cell execution and live plots in your browser.
2. **No Free-Tier Limits**: Infinite runtime, zero timeouts, no 12-hour session disconnects.
3. **No File Uploads/Downloads**: Directly accesses your local `dataset_rail-drishti/` and saves `RailDrishti.pt` straight to your `models/` folder.
4. **Hardware Acceleration**: Uses your Mac's **Apple Silicon GPU (MPS)** or Windows **NVIDIA CUDA GPU**.

---

## 🍎 macOS Setup (Step-by-Step)

### Step 1: Start the Local Backend Server in Terminal
Open your Mac Terminal in the `drishti-kavach` project root and run:

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Start the local server allowing Google Colab access (Single Command)
jupyter server \
  --ServerApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --ServerApp.port_retries=0 \
  --ServerApp.disable_check_xsrf=True
```

---

### Step 2: Copy the Local Connection Token
The terminal will display a line similar to:
```text
http://localhost:8888/?token=a1b2c3d4e5f67890abcdef...
```
👉 **Copy this full URL** (including `?token=...`).

---

### Step 3: Connect in Google Colab
1. In your browser, open [Google Colab](https://colab.research.google.com/).
2. Click **File $\rightarrow$ Upload notebook** $\rightarrow$ choose [`src/colab_training/train_on_colab.ipynb`](src/colab_training/train_on_colab.ipynb).
3. In the top-right corner, click the dropdown arrow next to the **Connect** button.
4. Select **"Connect to a local runtime"**.
5. Paste your copied `http://localhost:8888/?token=...` URL and click **Connect**.
6. Click **Runtime $\rightarrow$ Run all**!

---

## 🪟 Windows 10/11 Setup (Step-by-Step)

### Step 1: Start the Local Backend Server in PowerShell
Open **PowerShell** in the `drishti-kavach` project folder:

```powershell
# 1. Allow script execution & activate environment
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2. (Optional: If using NVIDIA GPU on Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Start the local server allowing Google Colab access
jupyter server --ServerApp.allow_origin='https://colab.research.google.com' --port=8888 --ServerApp.port_retries=0 --ServerApp.disable_check_xsrf=True
```

---

### Step 2: Copy the Local Connection Token
Look for the output in PowerShell:
```text
http://localhost:8888/?token=a1b2c3d4e5f67890abcdef...
```
👉 **Copy this full URL** (including `?token=...`).

---

### Step 3: Connect in Google Colab
1. In your browser, open [Google Colab](https://colab.research.google.com/).
2. Click **File $\rightarrow$ Upload notebook** $\rightarrow$ upload [`src/colab_training/train_on_colab.ipynb`](src/colab_training/train_on_colab.ipynb).
3. In the top-right corner, click the dropdown arrow next to the **Connect** button.
4. Select **"Connect to a local runtime"**.
5. Paste your copied `http://localhost:8888/?token=...` URL and click **Connect**.
6. Click **Runtime $\rightarrow$ Run all**!

---

## 📊 Live Monitoring & What Happens During Training

* **Live Red-to-Green Accuracy Card**: After every epoch, Colab outputs a minimal color-coded accuracy card comparing your model against Indian Railways field deployment targets.
* **Auto-Saved Weights**: When training finishes, the best weights are automatically saved to `models/RailDrishti.pt` on your local hard drive!

---

## 🛑 How to Stop / Disconnect
When you are done training:
* In your terminal or PowerShell, press **`[Ctrl + C]`** twice to shut down the local server.
