import pandas as pd
import os
import numpy as np

def calculate_sem(group):
    """Calculate the standard error of the mean for each column."""
    return group.std() / np.sqrt(len(group))

def process_individual_file(input_file, subject_code):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Check if required columns are present
    if not all(col in df.columns for col in ['ZT', 'wake_percent', 'non_rem_percent', 'rem_percent']):
        print(f"Error: File {input_file} is missing required columns.")
        return None

    # Group by zeitgeber_time and calculate mean and SEM for percentages
    result_df = df.groupby('ZT').agg(
        wake_percent_mean=('wake_percent', 'mean'),
        non_rem_percent_mean=('non_rem_percent', 'mean'),
        rem_percent_mean=('rem_percent', 'mean'),
        
        wake_percent_sem=('wake_percent', calculate_sem),
        non_rem_percent_sem=('non_rem_percent', calculate_sem),
        rem_percent_sem=('rem_percent', calculate_sem)
    ).reset_index()

    # Add the subject code column
    result_df['subject'] = subject_code

    return result_df

def process_multiple_files(input_files, output_file):
    combined_results = pd.DataFrame()

    for input_file in input_files:
        # Prompt the user to enter a subject code for each file
        subject_code = input(f"Enter subject code for {input_file}: ")
        
        # Process the individual file and get the results
        result_df = process_individual_file(input_file, subject_code)

        if result_df is not None:
            # Append each result_df to the combined_results DataFrame
            combined_results = pd.concat([combined_results, result_df], ignore_index=True)

    # Save the combined results to CSV
    combined_results.to_csv(output_file, index=False)
    print(f"Combined processed sleep data saved to: {output_file}")

# Get input files and output file paths
input_files = input("Enter paths of CSV files (comma-separated): ").split(',')
input_files = [file.strip() for file in input_files if os.path.isfile(file)]

if not input_files:
    print("No valid input files provided.")
else:
    output_file = input("Enter output CSV file path: ")
    if not os.path.dirname(output_file):
        print("Invalid output directory.")
    else:
        process_multiple_files(input_files, output_file)
