#!/bin/bash

# Add "# TensorRT Environment Variables" to the .bashrc file
echo "# TensorRT Environment Variables" >> ~/.bashrc

# Append TensorRT environment variables to the .bashrc file
TENSORRT_PATH=/opt/tensorrt
echo "export PATH=\$PATH:$TENSORRT_PATH/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$TENSORRT_PATH/lib" >> ~/.bashrc

# Add /opt/tensorrt/lib/ to the ldconfig configuration
echo "$TENSORRT_PATH/lib" | sudo tee /etc/ld.so.conf.d/tensorrt.conf

# Source the .bashrc file
source ~/.bashrc