# FastDemo — Multi-threaded Video Denoising on Jetson Orin NX

> Developed at **ENERZAi** (2024). Follow-up to [EasyDemo](https://github.com/bmycheez/EasyDemo).

Benchmark and reference implementation of **single-thread vs multi-thread** video denoising pipelines on **NVIDIA Jetson Orin NX**, in both C++ and Python.

## 📂 Implementations
| Folder | Language | Task |
|---|---|---|
| `cpp_rgb2rgb_3dnr` | C++ | RGB → RGB, 3D (temporal) denoising |
| `python_rgb2rgb_3dnr` | Python | RGB → RGB, 3D (temporal) denoising |
| `python_raw2raw_2dnr` | Python | RAW → RAW, 2D (spatial) denoising |

## 🔧 Setup (Jetson Orin NX)
CUDA and cuDNN are installed automatically with JetPack.

1. **OpenCV with CUDA (C++)** — [guide](https://forums.developer.nvidia.com/t/best-way-to-install-opencv-with-cuda-on-jetpack-5-xavier-nx-opencv-for-tegra/222777)
2. **Python virtual env** — [guide](https://jamanbbo.tistory.com/45)
3. **PyTorch for Jetson** — [guide](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
4. **TensorRT & PyCUDA** — [guide](https://medium.com/dropout-analytics/pycuda-on-jetson-nano-7990decab299)

## Tech
C++ · CMake · Python · TensorRT · OpenCV (CUDA) · multi-threading
