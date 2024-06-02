import re
import csv
import os
import glob

def write_data_to_csv(data, output_file):
    if data:
        # Sort the data by 'delete_batch_size' in ascending order before writing to the CSV
        sorted_data = sorted(data, key=lambda x: x.get('delete_batch_size', 0))
        keys = sorted_data[0].keys()  # Get all the keys from the first dictionary as headers
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for record in sorted_data:
                writer.writerow(record)

def parse_log_file(filepath):
    data = []
    current_record = {}

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()  # Strip the line here
            if not line:
                continue  # Skip empty lines
            
            if "filename:" in line:
                if current_record:  # Save the previous block before starting a new one
                    data.append(current_record)
                    current_record = {}
                current_record['filename'] = line.split(":")[1].strip()
            
            elif "Delete batch Size:" in line:
                current_record['delete_batch_size'] = int(line.split(":")[1].strip())

            elif "Depth of graph:" in line:
                current_record['depth_of_graph'] = int(line.split(":")[1].strip())

            elif line.startswith("Time of"):
                function_name = line.split(":")[0].split("Time of")[1].strip()
                function_time = float(line.split(":")[1].strip().split(" ")[0])
                current_record[function_name] = function_time

    if current_record:  # Don't forget to add the last block
        data.append(current_record)

    return data

# Usage example
log_files = glob.glob('log_file/*.log')

# Loop through all log files and process them
for log_file in log_files:
    parsed_data = parse_log_file(log_file)
    
    # Generate the output CSV filename based on the log file name
    output_csv_path = os.path.splitext(log_file)[0] + '_output_data.csv'
    
    # Write the parsed data to the corresponding CSV file
    write_data_to_csv(parsed_data, output_csv_path)
    print(f"Data from {log_file} has been written to {output_csv_path}")
