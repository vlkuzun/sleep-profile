
import pandas as pd
import os

# Function to calculate NREM packets
def calculate_nrem_packets(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Ensure the required columns exist
    if 'sleepStage' not in df.columns or 'Timestamp' not in df.columns:
        raise ValueError("The input CSV must contain 'sleepStage' and 'Timestamp' columns.")

    # Convert Timestamp to datetime for accuracy (optional, if needed)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Add a new column 'NREMpacket' initialized with 0
    df['NREMpacket'] = 0

    # Find continuous runs of sleepStage = 2
    run_start = None

    for i, stage in enumerate(df['sleepStage']):
        if stage == 2:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_length = i - run_start
                if run_length >= 20:  # Minimum 20 seconds run
                    df.loc[run_start:i-1, 'NREMpacket'] = 1  # Mark up to i-1, not i
                run_start = None

    # Handle case where the last rows are part of a valid run
    if run_start is not None:
        run_length = len(df) - run_start
        if run_length >= 20:
            df.loc[run_start:, 'NREMpacket'] = 1

    # Return the updated DataFrame
    return df

# Function to calculate REM episodes
def calculate_rem_episodes(df):
    # Add a new column 'REMepisode' initialized with 0
    df['REMepisode'] = 0
    df['NREMepisode'] = 0  # Initialize NREMepisode column

    # Find continuous runs of sleepStage = 3 (REM)
    rem_runs = []
    run_start = None

    for i, stage in enumerate(df['sleepStage']):
        if stage == 3:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                rem_runs.append((run_start, i - 1))  # Mark the run up to i-1
                run_start = None

    # Handle case where the last rows are part of a valid REM run
    if run_start is not None:
        rem_runs.append((run_start, len(df) - 1))  # End with the last valid index

    # Merge REM runs if gaps are less than 40 seconds and no NREM (sleepStage = 2) in the gap
    merged_rem_runs = []
    previous_run = rem_runs[0]

    for current_run in rem_runs[1:]:
        gap_start = previous_run[1] + 1
        gap_end = current_run[0] - 1

        # Check if there's a sleepStage=2 (NREM) within the gap
        if (gap_end - gap_start + 1) < 40:
            # If gap contains sleepStage=2 (NREM), don't merge the runs
            if (df['sleepStage'][gap_start:gap_end + 1] == 2).any():
                # Mark NREMepisode as 1 for the gap
                df.loc[gap_start:gap_end, 'NREMepisode'] = 1
                merged_rem_runs.append(previous_run)
                previous_run = current_run
            else:
                # Merge the runs if no NREM (sleepStage=2) in the gap
                previous_run = (previous_run[0], current_run[1])
        else:
            merged_rem_runs.append(previous_run)
            previous_run = current_run

    # Add the last run
    merged_rem_runs.append(previous_run)

    # Mark REM episodes in the DataFrame
    for start, end in merged_rem_runs:
        df.loc[start:end, 'REMepisode'] = 1

    return df

# Function to calculate NREM episodes
def calculate_nrem_episodes(df):
    # Add a new column 'NREMepisode' initialized with 0 if not already done
    if 'NREMepisode' not in df.columns:
        df['NREMepisode'] = 0

    # Find continuous runs of NREMpacket = 1 within eligible rows
    nrem_runs = []
    run_start = None

    # Ensure that eligible_for_nrem is defined (e.g., no REMepisode)
    eligible_for_nrem = df['REMepisode'] == 0

    for i, (packet, eligible) in enumerate(zip(df['NREMpacket'], eligible_for_nrem)):
        if packet == 1 and eligible:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                nrem_runs.append((run_start, i - 1))  # Mark the run up to i-1
                run_start = None

    # Handle case where the last rows are part of a valid NREM run
    if run_start is not None:
        nrem_runs.append((run_start, len(df) - 1))  # End with the last valid index

    # Merge NREM runs if gaps are less than 40 seconds and no REMepisode in the gap
    merged_nrem_runs = []
    previous_run = nrem_runs[0] if nrem_runs else None

    for current_run in nrem_runs[1:]:
        gap_start = previous_run[1] + 1
        gap_end = current_run[0] - 1

        # Check if there's a REMepisode = 1 within the gap
        if (gap_end - gap_start + 1) < 40:
            # If there's any REMepisode = 1 in the gap, do not merge
            if (df['REMepisode'][gap_start:gap_end + 1] == 1).any():
                merged_nrem_runs.append(previous_run)
                previous_run = current_run
            else:
                # Merge the runs if no REMepisode in the gap
                previous_run = (previous_run[0], current_run[1])
        else:
            merged_nrem_runs.append(previous_run)
            previous_run = current_run

    # Add the last run
    if previous_run:
        merged_nrem_runs.append(previous_run)

    # Mark NREM episodes in the DataFrame
    for start, end in merged_nrem_runs:
        df.loc[start:end, 'NREMepisode'] = 1

    return df

# Function to calculate WAKE episodes
def calculate_wake_episodes(df):
    # Add a new column 'WAKEepisode' where both REMepisode and NREMepisode are 0
    df['WAKEepisode'] = ((df['REMepisode'] == 0) & (df['NREMepisode'] == 0)).astype(int)

    return df

# Function to create the sleepStageConsolidated column, remove old columns, and save to file
def consolidate_sleep_stages(df, output_file_name):
    # Create the 'sleepStageConsolidated' column based on the conditions
    df['sleepStageConsolidated'] = 0
    df.loc[df['WAKEepisode'] == 1, 'sleepStageConsolidated'] = 1
    df.loc[df['NREMepisode'] == 1, 'sleepStageConsolidated'] = 2
    df.loc[df['REMepisode'] == 1, 'sleepStageConsolidated'] = 3

    # Remove the 'WAKEepisode', 'NREMepisode', and 'REMepisode' columns
    df.drop(columns=['NREMpacket', 'WAKEepisode', 'NREMepisode', 'REMepisode'], inplace=True)

    # Save the updated DataFrame to the output file with the provided name
    df.to_csv(output_file_name, index=False)

    return df  # Optionally return the updated dataframe

# Define input and output file paths
file_path = "/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz.csv"
output_file_name = "/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz_consolidated.csv"

# Calculate NREM packets
df = calculate_nrem_packets(file_path)

# Calculate REM episodes
df = calculate_rem_episodes(df)

# Calculate NREM episodes
df = calculate_nrem_episodes(df)

# Calculate WAKE episodes
df = calculate_wake_episodes(df)

# Consolidate sleep stages and save to file
consolidated_df = consolidate_sleep_stages(df, output_file_name)

print(f"Consolidated data saved to {output_file_name}")