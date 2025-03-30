import pandas as pd
import numpy as np
import datetime

def add_timestamp_to_csv(input_csv, output_csv, sampling_rate, start_time):
    """
    Adds a timestamp column to an existing CSV file based on the sampling rate and start time.
    
    Parameters:
    - input_csv: str, path to the input CSV file.
    - output_csv: str, path to the output CSV file where the modified CSV will be saved.
    - sampling_rate: int, the number of samples per second (Hz).
    - start_time: datetime, the starting timestamp for the first sample.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Get the number of rows (samples) in the DataFrame
    total_samples = len(df)
    
    # Generate a list of timestamps based on the start_time and sampling_rate
    time_deltas = pd.to_timedelta(np.arange(0, total_samples) / sampling_rate, unit='s')
    timestamps = start_time + time_deltas
    
    # Add the Timestamp column to the DataFrame
    df['Timestamp'] = timestamps
    
    # Ensure the Timestamp column is formatted as a string in the desired format
    df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    
    print(f"Timestamp column added and CSV saved to: {output_csv}")

if __name__ == "__main__":
    input_csv = input("Enter the path to the input CSV file without quotes: ")
    output_csv = input("Enter the path to save the output CSV file without quotes: ")
    sampling_rate = int(input("Enter the sampling rate in Hz (e.g., 512): "))
    start_time_str = input("Enter the start time (YYYY-MM-DD HH:MM:SS): ")
    
    # Convert the start time string to a datetime object
    start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    
    # Call the function with user-provided parameters
    add_timestamp_to_csv(input_csv, output_csv, sampling_rate, start_time)

# Example inputs:
# input_csv = "Y:/somnotate/to_score_set/test_auto_annotation/automated_state_annotationoutput_sub-012_ses-01_recording-01.csv"
# output_csv = "Y:/somnotate/to_score_set/test_auto_annotation/automated_state_annotationoutput_sub-012_ses-01_recording-01_timestamped.csv"
# sampling_rate = 512
# start_time_str = 2024-09-16 16:01:49


