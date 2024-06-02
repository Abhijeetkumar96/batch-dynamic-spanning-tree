import re
import csv
import os
import glob

def write_data_to_csv(data, output_file):
    if data:
        keys = data[0].keys()  # Get all the keys from the first dictionary as headers
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for record in data:
                writer.writerow(record)

def parse_log_file(filepath):
    data = []
    current_record = {}
    function_times = False

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()  # Strip the line here
            if not line:
                continue  # Skip empty lines
            
            if line.startswith("filename:"):
                if current_record:  # Save the previous block before starting a new one
                    data.append(current_record)
                    current_record = {}
                current_record['filename'] = line.split(":")[1].strip()
            
            elif "Deleted" in line:
                batch_size_match = re.search(r"Deleted : (\d+)", line)
                if batch_size_match:
                    current_record['Batch Size'] = int(batch_size_match.group(1))
            
            elif "Time (total):" in line:
                total_time_match = re.search(r"Time \(total\): ([\d.]+) ms", line)
                if total_time_match:
                    current_record['total_time'] = float(total_time_match.group(1))

    if current_record:  # Don't forget to add the last block
        data.append(current_record)

    return data

# Usage example
log_files = glob.glob('*.log')

# Loop through all log files and process them
for log_file in log_files:
    parsed_data = parse_log_file(log_file)

    # Sort the data by 'Batch Size'
    sorted_data = sorted(parsed_data, key=lambda x: x['Batch Size'])
    
    # Generate the output CSV filename based on the log file name
    output_csv_path = os.path.splitext(log_file)[0] + '_output_data.csv'
    
    # Write the sorted data to the corresponding CSV file
    write_data_to_csv(sorted_data, output_csv_path)
    print(f"Data from {log_file} has been written to {output_csv_path}")