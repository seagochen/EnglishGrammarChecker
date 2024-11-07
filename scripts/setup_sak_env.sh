#!/bin/bash

# Download the SAK repository from Github
git clone https://github.com/seagochen/sak.git 

# Move the SAK repository to the /opt directory
sudo mv sak /opt

# Append SAK environment variables to the .bashrc file
SAK_PATH=/opt/sak
echo "export PATH=\$PATH:$SAK_PATH/bin" >> ~/.bashrc

# Call the install_packages.sh script to set up the SAK environment
$SAK_PATH/environments/install_packages.sh

# Source the .bashrc file
source ~/.bashrc