import pandas as pd
import numpy as np
import os

def convert_to_zeitgeber_time(timestamp, lights_on_time='09:00', lights_off_time='21:00'):
    """
    Converts a timestamp to Zeitgeber Time (ZT), based on specific lights-on and lights-off times.
    The ZT will be calculated as a fractional hour (e.g., 6.5 for 15:30).
    Drop the first row to minimise the effect of headset implantation stress.
    
    Parameters:
    - timestamp (datetime): Original timestamp.
    - lights_on_time (str): Time representing ZT0 in HH:MM format (default '09:00').
    - lights_off_time (str): Time representing ZT12 in HH:MM format (default '21:00').
    
    Returns:
    - float: Zeitgeber time (ZT0-ZT23) in fractional hours.
    """
    lights_on = pd.to_datetime(lights_on_time).time()
    # Calculate the difference between the timestamp and lights_on in minutes
    minutes_since_lights_on = (timestamp - pd.to_datetime(f"2000-01-01 {lights_on_time}")).total_seconds() / 60
    # Convert to ZT (scaled to the 24-hour clock, where ZT0 = lights_on_time)
    zeitgeber_time = (minutes_since_lights_on / 60) % 24  # Convert minutes to fractional hours
    return zeitgeber_time

def process_sleep_data(input_file, output_file, bin_size, lights_on_time='09:00', lights_off_time='21:00'):
    try:
        # Load CSV file
        df = pd.read_csv(input_file)

        # Ensure Timestamp is in datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Convert sleepStage column to closest integer
        df['sleepStage'] = np.round(df['sleepStage'])

        # Create Zeitgeber time based on the specified lights-on and lights-off times
        df['ZT'] = df['Timestamp'].apply(lambda x: convert_to_zeitgeber_time(x, lights_on_time, lights_off_time))

        # Create bins for the specified time interval (bin_size)
        df['time_bin'] = df['Timestamp'].dt.floor(bin_size)

        # Calculate the percentage of each sleep stage within each time bin
        wake_df = df[df['sleepStage'] == 1]
        non_rem_df = df[df['sleepStage'] == 2]
        rem_df = df[df['sleepStage'] == 3]

        wake_percent = wake_df.groupby('time_bin')['sleepStage'].count() / df.groupby('time_bin')['sleepStage'].count() * 100
        non_rem_percent = non_rem_df.groupby('time_bin')['sleepStage'].count() / df.groupby('time_bin')['sleepStage'].count() * 100
        rem_percent = rem_df.groupby('time_bin')['sleepStage'].count() / df.groupby('time_bin')['sleepStage'].count() * 100

        # Create the result DataFrame and include Zeitgeber time
        result_df = pd.DataFrame({
            'time_bin': wake_percent.index,
            'wake_percent': wake_percent.values,
            'non_rem_percent': non_rem_percent.values,
            'rem_percent': rem_percent.values
        })

        # Add Zeitgeber time for each time_bin in the result DataFrame
        result_df['ZT'] = result_df['time_bin'].apply(lambda x: convert_to_zeitgeber_time(x, lights_on_time, lights_off_time))

        # Drop the first row before saving
        result_df = result_df.iloc[1:]

        # Fill any empty cells with 0
        result_df = result_df.fillna(0)

        # Save results to CSV
        result_df.to_csv(output_file, index=False)
        print(f"Processed sleep data saved to: {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")


# Get input and output file paths
input_file = input("Enter input CSV file path: ")

# Validate input file
if not os.path.isfile(input_file):
    print("Invalid input file.")
else:
    output_file = input("Enter output CSV file path: ")
    if not os.path.dirname(output_file):
        print("Invalid output directory.")
    else:
        # Ask for the bin size (e.g., '1h', '30min', '15min')
        bin_size = input("Enter the bin size (e.g., '1h', '30min', '15min'): ")
        process_sleep_data(input_file, output_file, bin_size)
