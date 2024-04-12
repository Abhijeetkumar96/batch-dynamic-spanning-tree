#!/bin/bash

# Define the input and output directories
# input_path="/home/cs22s501/spanning_tree/batch-dynamic-spanning-tree/v2/datasets/connected_datasets/"
input_path="/raid/graphwork/datasets/new_graphs/csr_bin/"

# Array to store process IDs
pids=()

for file in "$input_path"*.egr; do
    echo "Processing $file..."

    # Start the process in the background and redirect its output to a log file
    ./dataset_creation "$file" &

    # Store the process ID of the background process
    pids+=($!)

    echo "Processing started for $file."
done

# Wait for all background processes to finish
for pid in ${pids[@]}; do
    wait $pid
done

echo "All processing completed."
