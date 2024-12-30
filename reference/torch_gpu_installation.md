If `torch.cuda.is_available()` returns `False`, it means PyTorch is not able to detect the GPU. Below are the detailed steps to troubleshoot and activate CUDA in your local `.venv` environment:

---

### **1. Check CUDA Installation**

1. **Verify NVIDIA GPU Driver:**

   - Ensure the GPU drivers are installed and up-to-date.
   - Check driver version:
     ```bash
     nvidia-smi
     ```
     If this command fails, download and install the latest NVIDIA drivers from the [official NVIDIA website](https://www.nvidia.com/drivers).

2. **Check CUDA Toolkit:**
   - Confirm that the CUDA Toolkit is installed and matches the version compatible with your PyTorch installation:
     ```bash
     nvcc --version
     ```
     If not installed, download it from the [NVIDIA CUDA website](https://developer.nvidia.com/cuda-downloads).

---

### **2. Install PyTorch with CUDA Support**

Your `.venv` might have a CPU-only version of PyTorch. To fix this:

1. **Activate the Virtual Environment:**

   ```bash
   source .venv/bin/activate  # Linux/macOS
   .\.venv\Scripts\activate   # Windows
   ```

2. **Uninstall CPU-only PyTorch:**

   ```bash
   pip uninstall torch
   ```

3. **Install PyTorch with CUDA:**
   - Visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) to find the appropriate installation command based on your CUDA version.
   - For example, if you are using CUDA 11.8:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

---

### **3. Verify PyTorch CUDA Installation**

After installation:

1. Run this script in the `.venv`:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))  # Check GPU name
   ```
2. It should return `True` for `cuda.is_available()` and display the GPU name.

---

### **4. Debug Common Issues**

1. **Mismatched CUDA Version:**

   - Ensure the PyTorch version and the installed CUDA Toolkit are compatible.
   - Refer to the [PyTorch compatibility table](https://pytorch.org/get-started/previous-versions/).

2. **Environment Issues:**

   - Make sure the virtual environment is correctly activated before running any Python script.

3. **Install `nvidia-pyindex` (optional):**
   - For better GPU runtime management:
     ```bash
     pip install nvidia-pyindex
     pip install nvidia-torch
     ```

---

### **5. (Optional) Use GPU Inside a Container**

If issues persist on your host system, consider using Docker with NVIDIA container support:

1. Install Docker and NVIDIA Container Toolkit.
2. Use a PyTorch Docker image with GPU support:
   ```bash
   docker run --gpus all -it --rm pytorch/pytorch:latest
   ```

---

Would you like additional help troubleshooting any of these steps?
