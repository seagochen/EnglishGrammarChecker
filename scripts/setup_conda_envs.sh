#!/bin/bash

# 确保 Conda 初始化
if ! command -v conda &> /dev/null; then
    echo "Conda 未安装或未加入 PATH，请检查。"
    exit 1
fi

set -e

# 备份 base 环境
echo "备份 base 环境..."
conda create --name base_backup --clone base

# 创建并激活 base_ml 环境
echo "创建并安装 base_ml 环境..."
conda create --name base_ml python=3.10 -y
conda install -n base_ml numpy pandas scikit-learn matplotlib seaborn scipy ipywidgets tqdm pillow opencv -y
conda install -n base_ml -c conda-forge jupyterlab -y

# 创建 base_torch 并安装 PyTorch
echo "创建 base_torch 环境..."
conda create --name base_torch --clone base_ml -y

echo "选择 PyTorch 版本："
echo "1. CPU"
echo "2. GPU - CUDA 11.8"
echo "3. GPU - CUDA 12.4"
read -p "请输入数字选项: " choice

case $choice in
  1)
    conda install -n base_torch pytorch torchvision torchaudio cpuonly -c pytorch -y
    ;;
  2)
    conda install -n base_torch pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    ;;
  3)
    conda install -n base_torch pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
    ;;
  *)
    echo "无效选项，退出..."
    exit 1
    ;;
esac

# 创建并安装 YOLO 环境
echo "创建 note_yolo 环境..."
conda create --name note_yolo --clone base_torch -y
conda install -n note_yolo -c conda-forge ultralytics -y

# 创建并安装 Transformers 环境
echo "创建 note_face 环境..."
conda create --name note_face --clone base_torch -y
conda install -n note_face -c huggingface -c conda-forge datasets -y

echo "所有环境已成功创建并安装完毕。"
