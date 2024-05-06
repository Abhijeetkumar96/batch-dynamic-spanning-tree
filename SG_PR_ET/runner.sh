#!/bin/bash

# Base directory where the files are located
base_dir="/raid/graphwork/datasets/new_graphs/csr_bin/"

# Target base directory
target_base_dir="/raid/graphwork/spanning_tree_datasets/maybe_connected/"

# Loop through all files in the base directory
for file in "$base_dir"/*; do
    # Extract the basename of the file without the extension
    filename=$(basename "$file")
    filename="${filename%.*}"

    # Define the target directory path
    target_dir="${target_base_dir}${filename}/"

    # Check if the target directory exists
    if [ -d "$target_dir" ]; then
        # Print only paths to .txt files in the target directory
        find "$target_dir" -type f -name '*.txt' -print
    else
        echo "Directory $target_dir does not exist."
    fi
done
