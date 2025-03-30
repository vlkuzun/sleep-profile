import pandas as pd

def convert_sleep_stage(csv_file, output_file=None):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if 'sleepStage' column exists
    if 'sleepStage' in df.columns:
        # Replace 0 values with 1 in 'sleepStage' column
        df['sleepStage'] = df['sleepStage'].replace(0, 1)
        print("Converted 'sleepStage' values of 0 to 1.")
    else:
        print("Error: 'sleepStage' column not found in the CSV.")
        return
    
    # Save the modified DataFrame to a new CSV file if output_file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Modified file saved as: {output_file}")

    return df  # Return the DataFrame in case further processing is needed

# Example usage
csv_file = "/ceph/harris/somnotate/to_score_set/vis_back_to_csv/annotations_visbrain_sub-007_ses-01_recording-01_data-sleepscore_time-0-70.5h.csv"  # Path to the input CSV file
output_file = "/ceph/harris/somnotate/to_score_set/vis_back_to_csv/annotations_visbrain_sub-007_ses-01_recording-01_data-sleepscore_time-0-70.5h_refine.csv"  # Path to the output CSV file (optional)
df = convert_sleep_stage(csv_file, output_file)
