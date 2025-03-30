import pandas as pd

def downsample_csv(input_file, output_file, target_sampling_rate, original_sampling_rate=512):
    """
    Downsample the CSV data to the specified target sampling rate.
    
    :param input_file: Path to the input CSV file
    :param output_file: Path to the output downsampled CSV file
    :param target_sampling_rate: Target sampling rate (in Hz)
    :param original_sampling_rate: Original sampling rate (in Hz), default is 512
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Calculate the downsampling factor
    downsampling_factor = original_sampling_rate // target_sampling_rate
    
    # Downsample by selecting every nth row
    df_downsampled = df.iloc[::downsampling_factor, :].reset_index(drop=True)
    
    # Save the downsampled DataFrame to a new CSV file
    df_downsampled.to_csv(output_file, index=False)
    print(f"Downsampled CSV saved to {output_file}")

# Example usage
input_file = "/ceph/harris/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_export(HBH)_timestamped.csv"  # Replace with your input CSV path
output_file = "/ceph/harris/volkan/somnotate/somnotate_performance/sub-010_ses-01_recording-01_export(HBH)_timestamped_sr-1hz.csv"  # Replace with your desired output path
target_sampling_rate = 1  # Example target sampling rate in Hz

# Downsample the CSV
downsample_csv(input_file, output_file, target_sampling_rate)
