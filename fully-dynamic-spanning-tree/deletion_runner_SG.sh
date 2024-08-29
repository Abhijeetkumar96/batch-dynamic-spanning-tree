#!/bin/bash

# Base directory where the original datasets are located
base_dir="/raid/graphwork/datasets/new_graphs/csr_bin"

# Target base directory for the insertion/deletion contents
# maybe_connected is for deletion
target_base_dir="/raid/graphwork/spanning_tree_datasets/maybe_connected/"

# Log directory (make sure this exists or create it)
log_dir="/home/graphwork/cs22s501/spanning_tree/batch-dynamic-spanning-tree/main_variants/Behaviourial_Analysis/SG_deletion_log_files"
err_dir="/home/graphwork/cs22s501/spanning_tree/batch-dynamic-spanning-tree/main_variants/Behaviourial_Analysis/SG_deletion_err_files"

mkdir -p "$log_dir"
mkdir -p "$err_dir"

# Maximum time allowed for each command (e.g., 5 minutes)
TIMEOUT_DURATION="5m"

# List of specific filenames to process (without extensions)
files_to_process=(
    "kron_g500-logn18"
    "higgs-twitter"
    "kron_g500-logn19"
    "kron_g500-logn20"
    "hollywood-2009"
    "kron_g500-logn21"
    "com-Orkut"
    "soc-LiveJournal1"
    "uk-2002"
    "arabic-2005"
    "GAP-road"
    "uk-2005"
    "europe_osm"
)

# Convert the list to a space-separated string for easy pattern matching
files_to_process_str="${files_to_process[@]}"

# Loop through all files in the base directory
for file in "$base_dir"/*; do
    # Extract the basename of the file without extension
    filename=$(basename "$file" .egr)

    # Check if the filename is in the list of files to process
    if [[ " ${files_to_process_str} " == *" ${filename} "* ]]; then
        # Define the target directory path using the basename without extension
        target_dir="${target_base_dir}${filename}/"

        # Define the target directory path using the basename without extension
        base_filename="${filename%.*}"

        # Check if the target directory exists
        if [ -d "$target_dir" ]; then
            # Find all .txt files in the target directory and execute a command for each
            find "$target_dir" -type f -name '*.txt' -print | while read txt_file; do
                # Define output log filename based on both the original file and txt file
                txt_basename=$(basename "$txt_file" .txt)
                file_out="${log_dir}/${base_filename}.log"
                error_file="${err_dir}/${base_filename}_err.log"

                echo -e "\nRunning command: build/dynamic_spanning_tree -i $file -b $txt_file -d 2 -r SG -p PR >> $file_out 2> $error_file"
                # Run the command with a timeout
                timeout $TIMEOUT_DURATION build/dynamic_spanning_tree -i "$file" -b "$txt_file" -d 2 -r SG -p PR >> "$file_out" 2>> "$error_file"

                # Check the exit status of the timeout command
                if [ $? -eq 124 ]; then
                    echo "Command timed out for file: $file with batch file: $txt_file" >> "$error_file"
                fi
            done
        else
            echo "Directory $target_dir does not exist."
        fi
    else
        echo "Skipping file $filename"
    fi
done
