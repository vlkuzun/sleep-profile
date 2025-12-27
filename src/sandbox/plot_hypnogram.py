import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_hypnogram(file_path, start_time, end_time):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Ensure 'Timestamp' column is in datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Convert start_time and end_time to datetime format
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    # Select the data within the specified time range
    mask = (df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)
    df_filtered = df[mask]
    
    # Define the sleep stages for the hypnogram
    sleep_stage_dict = {1: 'Awake', 2: 'NREM', 3: 'REM'}
    
    # Calculate Zeitgeber Time (ZT), where 09:00:00 is ZT=0
    reference_time = start_time.replace(hour=9, minute=0, second=0, microsecond=0)
    df_filtered['ZT'] = (df_filtered['Timestamp'] - reference_time).dt.total_seconds() / 3600  # Convert seconds to hours
    
    # Calculate the ZT for the start and end times
    start_ZT = (start_time - reference_time).total_seconds() / 3600
    end_ZT = (end_time - reference_time).total_seconds() / 3600

    # Plot the hypnogram
    plt.figure(figsize=(16, 3))
    plt.plot(df_filtered['ZT'], df_filtered['sleepStage'], color='black', linestyle='-')

    # Adjust y-axis limits to reduce space between ticks
    plt.ylim(0.8, 3.2)  # Reduced space between ticks for more compact labels

    # Customize the plot
    plt.yticks([1, 2, 3], ['Awake', 'NREM', 'REM'])
    plt.gca().invert_yaxis()  # Inverts the y-axis so 'Awake' is at the top
    plt.xlabel('Zeitgeber Time', fontsize=20)
    plt.ylabel('')
    plt.title('')
    plt.xticks(fontsize=20)
    plt.tick_params(axis='y', labelsize=20)

    # Set x-axis limits based on the ZT of the start and end times
    plt.xlim(start_ZT, end_ZT)

    # Remove top and right spines
    ax = plt.gca()  # Get the current axis
    ax.spines['top'].set_visible(False)  # Hide the top spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
create_hypnogram('/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz.csv', 
                 '2024-11-30 10:00:00', '2024-11-30 18:00:00')
