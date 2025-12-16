import pandas as pd
import numpy as np
from scipy.signal import decimate

def load_csv(file_path):
    """Load a CSV file and return a DataFrame."""
    df = pd.read_csv(file_path)
    print(f"Loaded file: {file_path}")
    return df

def verify_row_count(df1, df2):
    """Check if two dataframes have the same number of rows."""
    if len(df1) != len(df2):
        print(f"Row count mismatch: File 1 has {len(df1)} rows, File 2 has {len(df2)} rows.")
        return False
    print("Rows of signal and timestamps/sleepscore match successfully.")
    return True

def replace_sleepstage(eeg_emg_df, sleepstage_df):
    """Replace the empty sleepstage column in the EEG/EMG DataFrame with data from the sleepstage DataFrame."""
    eeg_emg_df['Timestamp'] = sleepstage_df['Timestamp']
    eeg_emg_df['sleepStage'] = sleepstage_df['sleepStage']
    print("SleepStage column replaced successfully.")
    return eeg_emg_df

def downsample_data(df, columns, scale):
    """Downsample the selected columns by a given scale."""
    downsampled_data = {}
    for col in columns:
        downsampled_data[col] = decimate(df[col].values, scale)
    downsampled_df = pd.DataFrame(downsampled_data)
    downsampled_df['Timestamp'] = df['Timestamp'].iloc[::scale].reset_index(drop=True)
    downsampled_df['sleepStage'] = df['sleepStage'].iloc[::scale].reset_index(drop=True)
    print(f"Data downsampled by a factor of {scale}.")
    return downsampled_df

def save_combined_file(df, directory, subject, session, recording, extra_info, scale):
    """Save the processed DataFrame to a CSV file with a specified filename format."""
    filename = f"{directory}/combined_somno_downsampled_{subject}_{session}_{recording}_{extra_info}_scale_{scale}.csv"
    df.to_csv(filename, index=False)
    print(f"File saved successfully as: {filename}")

def main():
    # Get user input for file paths and settings
    eeg_emg_file = input("Enter the path for the EEG/EMG CSV file: ")
    sleepstage_file = input("Enter the path for the Sleep Stage CSV file: ")
    output_directory = input("Enter the output directory: ")
    subject = input("Enter subject identifier: ")
    session = input("Enter session identifier: ")
    recording = input("Enter recording identifier: ")
    extra_info = input("Enter any extra information for the filename: ")
    sampling_rate = int(input("Enter the sampling rate in Hz (e.g., 512): "))
    downsample_scale = int(input("Enter the downsampling scale (e.g., 2 for half the rate): "))

    # Load data from CSVs
    eeg_emg_df = load_csv(eeg_emg_file)
    sleepstage_df = load_csv(sleepstage_file)

    # Verify row count
    if not verify_row_count(eeg_emg_df, sleepstage_df):
        print("Exiting due to row count mismatch.")
        return  # Stop if there is a mismatch

    # Replace the SleepStage column
    eeg_emg_df = replace_sleepstage(eeg_emg_df, sleepstage_df)

    # Select columns to downsample (e.g., EEG1, EEG2, EMG)
    signal_columns = ['EEG1', 'EEG2', 'EMG']
    downsampled_df = downsample_data(eeg_emg_df, signal_columns, downsample_scale)

    # Save the combined and downsampled file
    save_combined_file(downsampled_df, output_directory, subject, session, recording, extra_info, downsample_scale)

    print("Processing complete.")

if __name__ == "__main__":
    main()
