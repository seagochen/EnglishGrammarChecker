#!/bin/bash

# Go back to the home directory
cd ~

# Make a directory for Miniconda
mkdir -p ~/miniconda3

# Download the Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Install Miniconda
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Remove the Miniconda installer
rm ~/miniconda3/miniconda.sh

# Add Miniconda to the PATH
~/miniconda3/bin/conda init bash

# Source the .bashrc file
source ~/.bashrc