# for HS_PR_PR
import pandas as pd
import glob
import os

# Directory path where your CSV files are located
directory_path = '/Users/abhijitkumarsahu/spanning_tree/new_log_files/HS_PR_PR/csv/'

# Use glob to create a list of file paths for all CSV files in the directory
file_paths = glob.glob(directory_path + '*.csv')

# Define the desired column order
desired_order = [
    "filename", 
    "Batch Size", 
    "total_time", 
]

# Process each file individually
for file_path in file_paths:
    try:
        # Get the base filename without the directory path or file extension
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        # Read the current csv file
        df = pd.read_csv(file_path)

        # Identify missing columns and fill them with NaN values
        missing_cols = set(desired_order) - set(df.columns)
        for col in missing_cols:
            df[col] = pd.NA

        # Reorder the DataFrame according to the desired column order
        df = df[desired_order]

        # Group by 'delete_batch_size' and compute the mean of numerical columns and use the first for constant columns
        grouped_df = df.groupby('Batch Size').agg({
            'filename': 'first',  # Since 'filename' is constant within each group
            'total_time': 'mean'  # Average of 'PR_RST'
        }).reset_index()

        # Now, round all numerical columns to 2 decimal places
        grouped_df = grouped_df.round(2);

        # Reorder the columns in the desired order
        grouped_df = grouped_df[['filename', 'Batch Size', 'total_time']]
        
        # Construct the output path using the constant filename and appending '_average.csv'
        output_path = f'/Users/abhijitkumarsahu/spanning_tree/new_log_files/HS_PR_PR/average/{base_filename}_average.csv'
        
        # Save the grouped DataFrame to the new CSV file with the constructed filename
        grouped_df.to_csv(output_path, index=False)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
