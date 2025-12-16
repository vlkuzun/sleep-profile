import pandas as pd
import numpy as np
import os

def downsample_csv(input_file, output_file, sampling_rate):
    """
    Downsample a CSV file containing sleep stage and timestamp recordings.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        sampling_rate (int): Desired sampling rate.

    Returns:
        None
    """

    # Load the CSV file
    df = pd.read_csv(input_file)

    # Ensure Timestamp is in datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Set Timestamp as index
    df.set_index('Timestamp', inplace=True)

    # Resample data using mode
    df_resampled = df.resample(f'{1000 // sampling_rate}ms').apply(
        lambda x: x.mode().iloc[0] if not x.empty else np.nan
    )

    # Reset index
    df_resampled.reset_index(inplace=True)

    # Save to CSV
    df_resampled.to_csv(output_file, index=False)


def main():
    # Get input file path from user
    input_file = input("Enter the input CSV file path: ")

    # Validate input file path
    while not os.path.isfile(input_file):
        print("Invalid file path. Please try again.")
        input_file = input("Enter the input CSV file path: ")

    # Get output directory path from user
    output_dir = input("Enter the output directory path: ")

    # Validate output directory path
    while not os.path.isdir(output_dir):
        print("Invalid directory path. Please try again.")
        output_dir = input("Enter the output directory path: ")

    # Get desired sampling rate from user
    sampling_rate = input("Enter the desired sampling rate (e.g., 128, 256): ")

    # Validate sampling rate
    while not sampling_rate.isdigit() or int(sampling_rate) <= 0:
        print("Invalid sampling rate. Please try again.")
        sampling_rate = input("Enter the desired sampling rate (e.g., 128, 256): ")

    # Get output file naming preferences
    print("Select output file naming format:")
    print("1. Sub-Ses-Recording-ExtraInfo")
    print("2. Custom")
    choice = input("Enter choice (1/2): ")

    if choice == "1":
        # Get subject, session, and recording info
        sub = input("Enter subject ID (e.g., sub-001): ")
        ses = input("Enter session ID (e.g., ses-01): ")
        rec = input("Enter recording ID (e.g., rec-01): ")
        extra = input("Enter extra info: ")

        # Derive output file path
        output_file = os.path.join(output_dir, f"{sub}_{ses}_{rec}_{extra}_sr-{sampling_rate}hz.csv")
    elif choice == "2":
        # Get custom output file name
        output_file_name = input("Enter custom output file name: ")
        output_file = os.path.join(output_dir, f"{output_file_name}.csv")
    else:
        print("Invalid choice. Using default naming format.")
        output_file = os.path.join(output_dir, f"downsampled_{os.path.basename(input_file)}")

    # Call downsampling function
    downsample_csv(input_file, output_file, int(sampling_rate))

    print(f"Downsampled CSV saved to: {output_file}")


if __name__ == "__main__":
    main()