#!/bin/bash

# Base directory where the original datasets are located
base_dir="/raid/graphwork/datasets/new_graphs/csr_bin"

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

# Output tar file
output_tar_file="/home/graphwork/cs22s501/spanning_tree/batch-dynamic-spanning-tree/main_variants/Behaviourial_Analysis/dataset_files.tar.xz"  # Update this path as needed

# Create a temporary directory to hold the files to be archived
temp_dir=$(mktemp -d)

# Loop through all files in the base directory
for file in "$base_dir"/*; do
    # Extract the basename of the file without extension
    filename=$(basename "$file" .egr)

    # Check if the filename is in the list of files to process
    if [[ " ${files_to_process_str} " == *" ${filename} "* ]]; then
        # Copy the file to the temporary directory
        cp "$file" "$temp_dir/"
    else
        echo "Skipping file $filename"
    fi
done

# Create the tar archive and compress it
tar -cvf - -C "$temp_dir" . | xz -z -k -v - > "$output_tar_file"

# Remove the temporary directory
rm -rf "$temp_dir"

echo "Files have been compressed into $output_tar_file"
