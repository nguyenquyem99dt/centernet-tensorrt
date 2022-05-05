# Overview
Convert ctdet CenterNet model to TensorRT engine.

# Enviroments
Tested on:
- Ubuntu 20.04 <br>
- Python: 3.7 <br>
- TensorRT: 7.0.0.11 <br>
- CMake: 3.16.3
- CUDA: 10.2
- CUB: 1.8.0

# Installation
## 1. Download and install TensorRT:
Download TensorRT 7.0.0.11 tar file from: https://developer.nvidia.com/nvidia-tensorrt-7x-download <br>
Expected location: /usr/local/TensorRT-7.0.0.11 <br>

## 2. Download and install CUB:
Download from: https://nvlabs.github.io/cub/ <br>
Expected location: /usr/local/cub-1.8.0

## 3. Clone this repo and install requirements
```bash
git clone https://github.com/nguyenquyem99dt/centernet-tensorrt.git
cd centernet-tensorrt
pip3 install -r requirements.txt
```
## 4. Build and install DCNv2 for CenterNet
```bash
cd CenterNet/src/lib/models/networks/dcn/
python3 setup.py build_ext --inplace
```
## 5. Clone TensorRT repo and replace neccessary files with custom files
Back to centernet-tensorrt root dir.
```bash
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT && git checkout release/7.0
git submodule update --init --recursive
cp -r ../custom-src/DCNv2 plugin/
cp ../custom-src/InferPlugin.cpp plugin/
cp ../custom-src/normalizePlugin.cpp plugin/normalizePlugin/
cp ../custom-src/CMakeLists.txt plugin/
```
## 6. Build TensorRT with custom plugin
```bash
mkdir build && cd build
cmake .. -DTRT_LIB_DIR=/usr/local/TensorRT-7.0.0.11/lib -DTRT_BIN_DIR=../out -DBUILD_PARSERS=OFF -DBUILD_SAMPLES=OFF -DBUILD_PLUGINS=ON -DCUB_ROOT_DIR=/usr/local/cub-1.8.0
make -j$(nproc)
```
After build, copy all files in "out" folder to TensorRT-7:
```bash
cp out/* /usr/local/TensorRT-7.0.0.11/lib/
```
## 7. Clone and build onnx-tensorrt
Back to centernet-tensorrt root dir.
```bash
git clone https://github.com/onnx/onnx-tensorrt.git
cd onnx-tensorrt && git checkout 7.0
git submodule update --init --recursive
cp ../custom-src/builtin_op_importers* ./
mkdir build && cd build
cmake .. -DTENSORRT_INCLUDE_DIR=/usr/local/TensorRT-7.0.0.11/include -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 -DTENSORRT_ROOT=/usr/local/TensorRT-7.0.0.11 -DTENSORRT_LIBRARY_INFER=/usr/local/TensorRT-7.0.0.11/lib/libnvinfer.so -DTENSORRT_LIBRARY_INFER_PLUGIN=/usr/local/TensorRT-7.0.0.11/lib/libnvinfer_plugin.so -DTENSORRT_LIBRARY_MYELIN=/usr/local/TensorRT-7.0.0.11/lib/libmyelin.so
make -j$(nproc)
```
## 8. Convert Pytorch model to TensorRT engine
Download pretrained weights from https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view and put it into weights/ directory. <br>
Run this command to covert model:
```bash
python3 run.py
```
