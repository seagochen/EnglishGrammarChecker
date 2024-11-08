#!/bin/bash

# Check if path input is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path>"
  exit 1
fi

# Echo the path and confirm the user input
echo "The path is: $1"
read -p "Is this the correct path? (y/n): " confirm

# Check if the user input is correct
if [ "$confirm" != "y" ]; then
  echo "The path is incorrect."
  exit 1
fi

# Check if the directory exists
if [ -d "$1" ]; then
  # Clean the common directory
  rm -rf "$1/common"

  # Copy the common directory to the specified path
  cp -r ./common "$1"
  echo "The common directory has been copied to the specified path."
else
  echo "The directory does not exist."
fi

# Exit the script
exit 0
