import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import os

def plot_sleep_stages(csv_file, output_dir, subject, session, recording, extra_info):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure 'Timestamp' is in datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    
    # Create rolling 3-hour windows and calculate sleep stage percentages in the previous 3 hours
    rolling_counts = df['sleepStage'].groupby(pd.Grouper(freq='3H', label='right')).value_counts().unstack(fill_value=0)
    rolling_percentages = rolling_counts.div(rolling_counts.sum(axis=1), axis=0) * 100
    
    # Reset index to get the Timestamp back as a column for plotting
    rolling_percentages = rolling_percentages.reset_index()

    # Create the plot with increased figure height for better label fitting
    fig, ax = plt.subplots(figsize=(15, 12))  # Increased height (9 instead of 6)

    # Plot each sleep stage as a line representing percentage in each prior 3-hour period
    time_bins = rolling_percentages['Timestamp']
    ax.plot(time_bins, rolling_percentages.get(1, 0), label='Wake (1)', color='orange')
    ax.plot(time_bins, rolling_percentages.get(2, 0), label='Non-REM (2)', color='blue')
    ax.plot(time_bins, rolling_percentages.get(3, 0), label='REM (3)', color='green')

    # Shade the background according to the dark phase (21:00-09:00) based on Timestamp
    start_date = rolling_percentages['Timestamp'].min().date()
    end_date = rolling_percentages['Timestamp'].max().date()
    current_date = start_date
    
    while current_date <= end_date:
        start_time = pd.to_datetime(f"{current_date} 21:00:00")
        end_time = pd.to_datetime(f"{current_date + timedelta(days=1)} 09:00:00")
        
        if start_time < rolling_percentages['Timestamp'].max() and end_time > rolling_percentages['Timestamp'].min():
            ax.axvspan(start_time, end_time, color='gray', alpha=0.5)
        
        current_date += timedelta(days=1)

    # Formatting the plot
    ax.set_title(f'{subject} - Sleep Stage Percentages in 3-Hour Periods')
    ax.set_xlabel('Time')
    ax.set_ylabel('Percentage')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.legend()
    plt.xticks(rotation=45)
    
    # Disable gridlines
    ax.grid(False)

    # Save the figure with the specified details
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/{subject}_{session}_{recording}_{extra_info}_sleep_stage_percentages_line_chart.png"
    plt.savefig(file_name)
    plt.close(fig)
    
    print(f"Figure saved to {file_name}")

# Gather inputs from the user
csv_file_path = input("Enter the path of your CSV file: ")
output_directory = input("Enter the output directory for the line chart: ")
subject = input("Enter subject: ")
session = input("Enter session: ")
recording = input("Enter recording: ")
extra_info = input("Enter extra information: ")

# Generate and save the plot
plot_sleep_stages(csv_file_path, output_directory, subject, session, recording, extra_info)