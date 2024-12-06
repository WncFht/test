---
categories: graph
date: 2024-11-15T10:51:06+0800
dir: graph
modify: 2024-12-06T00:14:44+0800
share: true
tags:
  - graph
title: 如何使用 docker
---

```thinking
让我思考一下如何�理解和分解这个问题...

首先,FFB6D是一个关于6D姿态估计的深度学习项目,发表在CVPR 2021。用Docker复现论文实验是一个很好的选择,可以确保环境的一致性。

我需要考虑几个关键点:
1. 环境依赖 - FFB6D使用PyTorch和CUDA,需要确保Docker镜像包含正确版本
2. 数据集处理 - 论文使用了LineMOD等数据集
3. 代码结构 - 需要考虑如何组织Dockerfile
4. 潜在问题 - GPU支持、数据挂载等可能遇到的问题

让我想一下用户可能的需求和痛点:
- 可能是第一次使用Docker
- 希望能快速开始实验
- 需要详细的步骤指导
- 可能会遇到环境问题

我应该提供一个循序渐进的指导:
- 先介绍基本概念
- 然后是环境准备
- 接着是具体步骤
- 最后是常见问题解决

我还需要确保:
- 指令准确且易于理解
- 充分说明每个步骤的目的
- 提供错误处理的建议
```

我来帮你完整地讲解如何用 Docker 复现 FFB6D 论文。让我们分步骤来进行:

1. 环境准备
- 首先确保你的系统已安装:
  - Docker (建议使用最新稳定版)
  - NVIDIA 驱动 (支持 CUDA 10.2 或更高)
  - nvidia-docker2 (用于 GPU 支持)

2. 创建 Dockerfile:

```dockerfile
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 克隆项目
RUN git clone https://github.com/ethnhe/FFB6D.git /ffb6d
WORKDIR /ffb6d

# 安装Python依赖
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.7.1 torchvision==0.8.2

# 编译CUDA操作
RUN cd ./ffb6d/models/RandLA/
RUN python3 setup.py install
```

3. 构建镜像：

```bash
docker build -t ffb6d .
```

4. 运行容器：

```bash
docker run --gpus all -it \
  -v /path/to/your/dataset:/ffb6d/datasets \
  -v /path/to/your/weights:/ffb6d/train_log \
  ffb6d /bin/bash
```

5. 数据准备：
- 下载 LineMOD 数据集
- 按照项目要求组织数据目录结构
- 运行数据预处理脚本

6. 训练和测试：

```bash
# 在容器内
cd /ffb6d
python3 train_custom.py --local_rank 0 --gpu '0' --num_threads 0
```

需要注意的几个关键点：

1. 数据集挂载
- 使用 Docker volume 挂载数据集和权重文件
- 确保路径正确且有适当的权限

2. GPU 支持
- 确保系统已正确安装 NVIDIA 驱动
- 使用 `--gpus all` 参数启用 GPU 支持

3. 常见问题解决：
- 如果遇到 CUDA 错误，检查显卡驱动版本是否兼容
- 内存不足时可以调整 batch size
- 权限问题可以通过调整挂载目录的权限解决

你需要我详细解释哪个部分吗？或者你在实践过程中遇到了什么具体问题？

```prompt
请你帮助我完成Docker (建议使用最新稳定版) * NVIDIA驱动 (支持CUDA 10.2或更高) * nvidia-docker2 (用于GPU支持)。 我用的是 Ubuntu 24.04
```

```
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
# 测试基本Docker功能
docker run hello-world

# 测试NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

1. 首先，让我们修复 NVIDIA 仓库配置。使用更直接的方式：

```bash
# 删除现有的配置
sudo rm /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 创建新的配置文件
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list << EOF
deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/ubuntu22.04/ubuntu22.04 main
EOF

# 更新GPG密钥
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

2. 扩展 Docker 配置，添加更多镜像和网络选项：

```bash
sudo tee /etc/docker/daemon.json << EOF
{
  "registry-mirrors": [                         
    "https://docker.1panel.dev",
  ],                                                    
    "dns": ["8.8.8.8", "8.8.4.4"],                      
    "max-concurrent-downloads": 10,                     
    "log-driver": "json-file",                          
    "log-opts": {                                       
        "max-size": "10m",                              
        "max-file": "3"                                 
    }                                                   
}                                                       
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
```

```
sudo tee /etc/docker/daemon.json << EOF
{
  "registry-mirrors": [
    "https://docker.1panel.dev",
    ]  
}                                                       
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
```

3. 检查网络配置：

```bash
# 测试DNS解析
dig registry-1.docker.io

# 测试网络连接
curl -v https://registry-1.docker.io/v2/

# 如果上面的命令失败，可以尝试临时使用代理
# 如果你有代理服务器，可以配置Docker使用代理：
sudo mkdir -p /etc/systemd/system/docker.service.d/
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf << EOF
[Service]
Environment="HTTP_PROXY=http://your-proxy:port"
Environment="HTTPS_PROXY=http://your-proxy:port"
EOF
```

完成这些配置后：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker

# 测试配置
docker info
docker pull hello-world
```

如果还是遇到网络问题，你能告诉我：

1. `ping registry-1.docker.io` 的结果是什么？
2. 你的网络环境是否使用代理？
3. `curl -v https://www.docker.com` 能否成功？

# 先尝试直接拉取基础镜像

```
docker pull nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
```

# 然后修改 Dockerfile 第一行为：

```
# 使用较新的 CUDA 镜像
FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace/code
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 3.8 为默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --set python3 /usr/bin/python3.8

# 升级 pip
RUN python3 -m pip install --upgrade pip

# 安装 PyTorch 和基础依赖
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他 Python 依赖
RUN pip3 install \
    opencv-python==4.4.0.46 \
    pytest-runner==5.3.1 \
    matplotlib==3.3.4 \
    tqdm==4.64.0 \
    tensorboard==2.10.1 \
    transforms3d==0.3.1 \
    open3d==0.9.0.0 \
    scipy==1.7.3 \
    numpy==1.19.5

# 复制预下载的文件
COPY downloads/apex /workspace/apex
COPY downloads/normalspeed /workspace/normalspeed
COPY code /workspace/code

# 安装 apex
RUN cd /workspace/apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# 安装 normalspeed
RUN cd /workspace/normalspeed && \
    python3 setup.py build_ext --inplace && \
    python3 setup.py install

WORKDIR /workspace/code
```

```
# 在 ffb6d_docker 目录下执行
cd ..
git clone https://github.com/ethnhe/FFB6D.git
cp -r FFB6D/* ffb6d_docker/
cd ffb6d_docker
```

```
nano run_ffb6d.sh
```

```run_ffb6d.sh
#!/bin/bash

# 创建必要的目录
mkdir -p data train_log

# 运行容器
docker run --gpus all -it \
    --shm-size=8g \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/data:/ffb6d/datasets \
    -v $(pwd)/train_log:/ffb6d/train_log \
    --name ffb6d_container \
    ffb6d:latest \
    bash
```

```
chmod +x run_ffb6d.sh
docker build -t ffb6d:latest .
# 创建数据目录
mkdir -p data train_log

# 设置目录权限
sudo chown -R $USER:$USER data train_log


```

```
./run_ffb6d.sh
# 测试 Python 环境
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 测试 CUDA 是否可用
nvidia-smi

# 检查 FFB6D 代码是否正确复制
ls -la
```

[另一个完整的构建](./%E5%8F%A6%E4%B8%80%E4%B8%AA%E5%AE%8C%E6%95%B4%E7%9A%84%E6%9E%84%E5%BB%BA.md)