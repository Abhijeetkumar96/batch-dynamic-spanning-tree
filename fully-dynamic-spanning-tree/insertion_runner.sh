#!/bin/bash

# Base directory where the original datasets are located
base_dir="/raid/graphwork/datasets/new_graphs/csr_bin"

# Target base directory for the insertion/deletion contents
# maybe_connected is for deletion
# insertion_datasets is for insertion
# target_base_dir="/raid/graphwork/spanning_tree_datasets/maybe_connected/"
target_base_dir="/raid/graphwork/spanning_tree_datasets/insertion_datasets/"

# Log directory (make sure this exists or create it)
log_dir="/home/graphwork/cs22s501/spanning_tree/batch-dynamic-spanning-tree/main_variants/Final_Version/insertion_log_files"
err_dir="/home/graphwork/cs22s501/spanning_tree/batch-dynamic-spanning-tree/main_variants/Final_Version/insertion_err_files"

mkdir -p "$log_dir"
mkdir -p "$err_dir"

# List of specific files you want to process
files=(
    kmer_V1r.egr
    kmer_V2a.egr
)

# Loop through all files in the base directory
# for file in "$base_dir"/*; do
for file_name in "${files[@]}"; do
    file="$base_dir/$file_name"
    # Extract the basename of the file
    filename=$(basename "$file")

    # Define the target directory path using the basename without extension
    base_filename="${filename%.*}"
    target_dir="${target_base_dir}${base_filename}/"

    # Check if the target directory exists
    if [ -d "$target_dir" ]; then
        # Find all .txt files in the target directory and execute a command for each
        find "$target_dir" -type f -name '*.txt' -print | while read txt_file; do
            # Define output log filename based on both the original file and txt file
            txt_basename=$(basename "$txt_file" .txt)
            file_out="${log_dir}/${base_filename}.log"
            error_file="${err_dir}/${base_filename}_err.log"

            echo "\nRunning command: build/dynamic_spanning_tree -i $file -b $txt_file -d 0 -r HS -p ET >> $file_out 2>> $error_file"
            # Uncomment the following line to actually run the command
            build/dynamic_spanning_tree -i "$file" -b "$txt_file" -d 0 -r HS -p ET >> "$file_out" 2>> "$error_file"
        done
    else
        echo "Directory $target_dir does not exist."
    fi
done
