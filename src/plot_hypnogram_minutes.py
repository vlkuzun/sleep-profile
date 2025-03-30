import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    
    # Calculate minutes from start time
    df_filtered['Minutes'] = (df_filtered['Timestamp'] - start_time).dt.total_seconds() / 60  # Convert seconds to minutes
    
    # Calculate the time in minutes for the start and end times
    start_minutes = 1  # Start the first tick at 1 minute, not 0
    end_minutes = (end_time - start_time).total_seconds() / 60  # End time in minutes from start time

    # Plot the hypnogram
    plt.figure(figsize=(16, 3))  # Adjusted figure height to compress the y-axis
    plt.plot(df_filtered['Minutes'], df_filtered['sleepStage'], color='black', linestyle='-')

    # Adjust y-axis limits to reduce space between ticks
    plt.ylim(0.8, 3.2)  # Reduced space between ticks for more compact labels

    # Use MaxNLocator to control the number of ticks and bring them closer
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower'))  # Only 1, 2, 3 on y-ticks

    # Customize the plot
    plt.yticks([1, 2, 3], ['Awake', 'NREM', 'REM'], fontsize=22)
    plt.gca().invert_yaxis()  # Inverts the y-axis so 'Awake' is at the top
    plt.xlabel('Time (minutes)', fontsize=22)
    plt.xticks(fontsize=22)

    # Set x-axis limits based on the Minutes of the start and end times
    plt.xlim(start_minutes, end_minutes)

    # Generate 6 x-ticks, with the first tick at 10 minutes and last tick at 60 minutes
    x_ticks = np.linspace(10, 90, 9)  # Generate 6 ticks starting from 10 to 60 minutes - alter as required
    plt.xticks(x_ticks, [f'{int(x)}' for x in x_ticks])  # Format x-ticks as integer minutes

    # Remove top and right spines
    ax.spines['top'].set_visible(False)  # Hide the top spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    
    # Show the plot
    plt.tight_layout()
    plt.show()


# Example usage
create_hypnogram('/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis_consolidated/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz_consolidated.csv', 
                 '2024-11-30 11:30:00', '2024-11-30 13:00:00')
