# GroundingDINO (Custom GPU-Optimized Version)

This repository contains a modified version of **GroundingDINO**, optimized to run on Windows systems with NVIDIA GPUs (especially RTX 40-series) that may have compatibility issues with the default installation.

It includes:
1.  **Fixed Setup**: Modified `setup.py` to bypass C++ extension compilation failures.
2.  **GPU Fallback**: Patched code to use PyTorch native functions if custom CUDA extensions fail, ensuring GPU acceleration still works.
3.  **Dependency Fixes**: Resolves conflicts with `transformers` and `torch`.

## 1. System Requirements

*   **OS**: Windows 10/11 (Tested on Windows).
*   **GPU**: NVIDIA GPU with CUDA support (Tested on RTX 4050).
*   **Python**: 3.8 - 3.11.
*   **CUDA Toolkit**: The project runs even if your system CUDA version (from `nvcc`) doesn't match the PyTorch CUDA version, thanks to the included fixes.

## 2. Environment Setup

It is highly recommended to use a virtual environment.

### Step 1: Create and Activate Environment
Open PowerShell and run:

```powershell
# Create venv
python -m venv venv

# Activate venv
.\venv\Scripts\Activate
```

### Step 2: Install PyTorch (CUDA 12.1)
We need a specific PyTorch version for modern GPUs.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Dependencies
Install the package in editable mode (this installs `transformers`, `opencv`, etc.):

```powershell
pip install -e .
```

*Note: If you see a "Failed to build wheel" error, ignore it as long as `pip list` shows `groundingdino` is installed. The custom `setup.py` handles the critical parts.*

### Step 4: Fix Transformers Version
The latest transformers library can cause errors. Downgrade to a stable version:

```powershell
pip install transformers==4.30.2
```

## 3. Pre-Run Checks

Before running inference, ensure you have:

1.  **Weights**: Download the model weights.
    *   Create a folder `weights/`
    *   Download [groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) into it.
2.  **Image**: Have an image ready (e.g., in `.asset/cat_dog.jpeg`).

## 4. How to Run

We have provided a simple script `run_inference.py` for easy usage.

1.  **Edit `run_inference.py`** to set your image path and prompt:
    ```python
    IMAGE_PATH = "path/to/your/image.jpg"
    TEXT_PROMPT = "cat . dog . person ."
    ```

2.  **Run the script**:
    ```powershell
    python run_inference.py
    ```

3.  **Check Output**: The result will be saved to `outputs/my_result.jpg`.

### Command Line Usage
Alternatively, use the original demo script:

```powershell
python demo/inference_on_a_image.py ^
  -c groundingdino/config/GroundingDINO_SwinT_OGC.py ^
  -p weights/groundingdino_swint_ogc.pth ^
  -i .asset/cat_dog.jpeg ^
  -o outputs ^
  -t "cat . dog ."
```

## 5. Common Errors & Fixes

| Error | Fix |
| :--- | :--- |
| `AttributeError: 'BertModel' ...` | Run `pip install transformers==4.30.2` |
| `Failed to load custom C++ ops` | **Normal.** This is a warning. The code will fallback to PyTorch native GPU functions. |
| `torch.cuda.is_available() -> False` | Ensure you installed PyTorch with the `--index-url .../cu121` flag. |
| `nvcc not found` | Ignore. Use the provided instructions; this setup doesn't require `nvcc` to be perfect. |

## 6. Project Structure

*   `groundingdino/`: Core model code (modified for compatibility).
*   `run_inference.py`: **Main entry point** for testing.
*   `weights/`: Store your `.pth` model weights here (Excluded from git).
*   `outputs/`: Generated images (Excluded from git).
