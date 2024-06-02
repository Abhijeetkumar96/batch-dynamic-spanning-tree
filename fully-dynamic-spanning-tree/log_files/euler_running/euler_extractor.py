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
    max_regular = 0
    min_my = float('inf')
    
    with open(filepath, 'r') as file:
        for line in file:
            # Extract filename
            if "filename:" in line:
                if current_record:  # If there's existing data, save it before starting a new record
                    current_record['max_regular'] = max_regular
                    current_record['min_my'] = min_my
                    data.append(current_record)

                filename_full_path = re.search(r"filename: (\S+)", line)
                if filename_full_path:
                    full_path = filename_full_path.group(1)
                    base_name = os.path.basename(full_path)
                    specific_name = os.path.splitext(base_name)[0]
                    current_record = {'filename': specific_name}
                    max_regular = 0  # Reset the maximum for Regular
                    min_my = float('inf')  # Reset the minimum for My

            elif "Regular eulerian Tour" in line:
                time = re.search(r"finished in: (\d+) us", line)
                if time:
                    max_regular = max(max_regular, int(time.group(1)))

            elif "My eulerian Tour" in line:
                time = re.search(r"finished in: (\d+) us", line)
                if time:
                    min_my = min(min_my, int(time.group(1)))

        # After the last line, save the last record
        if current_record:
            current_record['max_regular'] = max_regular
            current_record['min_my'] = min_my
            data.append(current_record)

    return data

# Usage example
# file_path = 'example.log'
# parsed_data = parse_log_file(file_path)
# print(parsed_data)

# Call the function to write data to CSV
# output_csv_path = 'output_data.csv'
# write_data_to_csv(parsed_data, output_csv_path)

# Find all .log files in the current directory
log_files = glob.glob('*.log')

# Loop through all log files and process them
for log_file in log_files:
    parsed_data = parse_log_file(log_file)
    
    # Generate the output CSV filename based on the log file name
    output_csv_path = os.path.splitext(log_file)[0] + '_output_data.csv'
    
    # Write the parsed data to the corresponding CSV file
    write_data_to_csv(parsed_data, output_csv_path)
    print(f"Data from {log_file} has been written to {output_csv_path}")
