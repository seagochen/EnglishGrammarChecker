#!/bin/bash

# 检查是否安装了 ultralytics 库
if ! python3 -c "import ultralytics" &> /dev/null; then
    echo "ultralytics 库未安装，正在安装..."
    pip install ultralytics
else
    echo "ultralytics 库已安装，跳过安装步骤。"
fi

# 使用 Python 下载 YOLOv8n 模型
python3 - <<EOF
from ultralytics import YOLO

# 加载 YOLOv8n 模型
model = YOLO('yolov8n.pt')  # 下载并加载 YOLOv8n 模型
print("YOLOv8n 模型下载完成并已加载")

# 加载 YOLOv8n-Pose 模型
model = YOLO('yolov8n-pose.pt')  # 下载并加载 YOLOv8n-pose 模型
print("YOLOv8n Pose 模型下载完成并已加载")
EOF

echo "脚本执行完毕。YOLOv8n 模型已下载并加载。"
