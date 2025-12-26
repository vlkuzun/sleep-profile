import pandas as pd

def stitch_csv_files(*input_csv_files):
    """
    Takes in multiple CSV files, concatenates their rows (excluding the headers of subsequent files),
    and writes the result to a user-specified output CSV file.

    Args:
        *input_csv_files: Paths of the input CSV files (in the desired order).

    Returns:
        None
    """
    # List to store dataframes from all input CSV files
    dataframes = []

    # Iterate over the provided CSV file paths
    for i, file in enumerate(input_csv_files):
        if i == 0:
            # Read the first CSV with the header
            df = pd.read_csv(file)
        else:
            # Read subsequent CSVs, skipping the first row (header), without modifying column titles
            df = pd.read_csv(file, skiprows=1, header=None)
            df.columns = pd.read_csv(file, nrows=0).columns  # Use the column names from the first file
        dataframes.append(df)

    # Concatenate all dataframes row-wise
    stitched_data = pd.concat(dataframes, ignore_index=True)

    # Ensure proper data types and remove stray commas
    stitched_data = stitched_data.applymap(lambda x: x.strip(',') if isinstance(x, str) else x)
    stitched_data = stitched_data.convert_dtypes()  # Ensure proper data types (e.g., int remains int)

    # Ask the user for the output file path
    output_csv_file = input("Enter the output CSV file path: ")

    # Write the concatenated data to the output CSV file, including the header from the first CSV
    stitched_data.to_csv(output_csv_file, index=False)
    print(f"Data has been stitched and saved to {output_csv_file}")

# Example usage:
stitch_csv_files("/Volumes/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-20h_50Hz.csv", 
                 "/Volumes/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-20-49h_50Hz.csv")
