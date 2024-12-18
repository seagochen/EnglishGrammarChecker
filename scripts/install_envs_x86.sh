#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Variables
ENV_NAME="dev_cuda"
PYTHON_VERSION="3.10"

# Function to print messages
print_message() {
    echo "=============================="
    echo "$1"
    echo "=============================="
}

# Function to update package list
update_packages() {
    print_message "Updating package list..."
    sudo apt update
}

# Function to install system dependencies
install_dependencies() {
    print_message "Installing OpenCV and other system dependencies..."
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        pkg-config \
        libgtk-3-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        gfortran \
        openexr \
        libatlas-base-dev \
        python3-dev \
        python3-numpy \
        libtbb2 \
        libtbb-dev \
        libdc1394-dev
}

# Function to install OpenCV
install_opencv() {
    print_message "Installing OpenCV..."
    sudo apt install -y libopencv-dev
}

# Function to install Protobuf
install_protobuf() {
    print_message "Installing Protobuf..."
    sudo apt install -y protobuf-compiler libprotobuf-dev
}

# Function to install YAML library
install_yaml() {
    print_message "Installing YAML library..."
    sudo apt install -y libyaml-cpp-dev
}

# Function to install Mosquitto
install_mosquitto() {
    print_message "Installing Mosquitto..."
    sudo apt install -y mosquitto mosquitto-clients libmosquitto-dev
}

# Function to check installed versions
check_versions() {
    print_message "Checking installed versions..."

    # OpenCV
    if pkg-config --exists opencv4; then
        opencv_version=$(pkg-config --modversion opencv4)
        echo "OpenCV version: $opencv_version"
    else
        echo "OpenCV is not installed."
    fi

    # Protobuf
    if command -v protoc &> /dev/null; then
        protobuf_version=$(protoc --version)
        echo "Protobuf version: $protobuf_version"
    else
        echo "Protobuf is not installed."
    fi

    # YAML
    yaml_version=$(dpkg -s libyaml-cpp-dev 2>/dev/null | grep '^Version:')
    if [ -n "$yaml_version" ]; then
        echo "YAML version: $yaml_version"
    else
        echo "YAML library is not installed."
    fi

    # Mosquitto
    mosquitto_version=$(mosquitto -h 2>/dev/null | grep -i version || echo "Mosquitto is not installed.")
    echo "Mosquitto version: $mosquitto_version"
}

# Function to create and set up conda environment
setup_conda_environment() {
    print_message "Setting up Conda environment: $ENV_NAME"

    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        echo "Conda is not installed. Please install Miniconda or Anaconda first."
        exit 1
    fi

    # Create a new environment
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

    # Install PyTorch with CUDA support
    conda install -n "$ENV_NAME" pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

    # Install Transformers from Hugging Face
    conda install -n "$ENV_NAME" -c huggingface transformers -y

    # Install Ultralytics from conda-forge
    conda install -n "$ENV_NAME" -c conda-forge ultralytics -y

    # Install additional Python packages
    conda install -n "$ENV_NAME" -c conda-forge \
        opencv \
        jupyterlab \
        scikit-learn \
        pandas \
        matplotlib \
        yfinance \
        plotly \
        paramiko \
        polygon-api-client -y
}

# Main execution flow
main() {
    update_packages
    install_dependencies
    install_opencv
    install_protobuf
    install_yaml
    install_mosquitto
    check_versions
    setup_conda_environment

    print_message "Installation and setup complete!"
    echo "To activate your Conda environment, run:"
    echo "    conda activate $ENV_NAME"
}

# Run the main function
main
