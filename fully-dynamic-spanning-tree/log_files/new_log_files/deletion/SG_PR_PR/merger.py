import pandas as pd
import glob
import os

# Directory path where your CSV files are located
directory_path = '/Users/abhijitkumarsahu/spanning_tree/new_log_files/deletion/SG_PR_PR/average/'

# Use glob to create a list of file paths for all CSV files in the directory
file_paths = glob.glob(directory_path + '*.csv')

# Initialize an empty DataFrame to store the concatenated data
all_data = pd.DataFrame()

# Read each CSV file and append it to the DataFrame
for file_path in file_paths:
    try:
        # Read the current csv file
        df = pd.read_csv(file_path)

        # Optionally, if you want to add a column to track the file origin
        df['source_file'] = os.path.basename(file_path)

        # Append the current DataFrame to the all_data DataFrame
        all_data = pd.concat([all_data, df], ignore_index=True)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Now you have a single DataFrame 'all_data' containing all the data from the CSV files
# If you want to reorder based on the columns present in 'all_data', you can do that before saving

# Save the concatenated DataFrame to a new CSV file
output_path = '/Users/abhijitkumarsahu/spanning_tree/new_log_files/deletion/SG_PR_PR/merged_file.csv'
all_data.to_csv(output_path, index=False)
