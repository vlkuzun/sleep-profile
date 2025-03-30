import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle

from scipy.ndimage import gaussian_filter1d

def plot_sleep_stages(files, subjects):
    """
    Plots wake_percent, non_rem_percent, and rem_percent from multiple files on the same graph.
    Adds a 12-hour cycle bar beneath the main graph, colored based on ZT_Adjusted values.

    Args:
        files (list of str): List of file paths. Each file should have columns ZT, wake_percent, non_rem_percent, and rem_percent.
        subjects (list of str): List of subject names corresponding to each file.
    """
    if not isinstance(files, list) or len(files) == 0:
        raise ValueError("Please provide a list of file paths.")

    if not isinstance(subjects, list) or len(subjects) != len(files):
        raise ValueError("Please provide a list of subject names matching the number of files.")

    # Define consistent colors for each sleep stage
    colors = {
        'wake_percent': 'red',
        'non_rem_percent': 'blue',
        'rem_percent': 'green',
    }

    # Define line styles for different files
    line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

    # Map stage names to user-friendly labels
    stage_labels = {
        'wake_percent': 'Awake',
        'non_rem_percent': 'NREM',
        'rem_percent': 'REM'
    }

    # Create the main plot
    fig, ax1 = plt.subplots(figsize=(16, 4))  # Increase the figure height for space for the bar

    min_zt_adjusted = float('inf')
    max_zt_adjusted = -float('inf')

    for idx, (file, subject) in enumerate(zip(files, subjects)):
        # Read the file into a DataFrame
        try:
            data = pd.read_csv(file)
        except Exception as e:
            print(f"Could not read file {file}: {e}")
            continue

        # Check if required columns exist
        required_columns = ['ZT', 'wake_percent', 'non_rem_percent', 'rem_percent']
        if not all(col in data.columns for col in required_columns):
            print(f"File {file} is missing required columns.")
            continue

        # Adjust ZT to create unique x-axis values for continuous cycles
        max_zt = 24
        unique_x_values = []
        last_zt = None
        cycle = 0

        for zt in data['ZT']:
            if last_zt is not None and zt < last_zt:
                cycle += 1
            unique_x_values.append(zt + cycle * max_zt)
            last_zt = zt

        data['ZT_Adjusted'] = unique_x_values

        # Update min and max ZT_Adjusted if data exists
        if not data['ZT_Adjusted'].empty:
            min_zt_adjusted = min(min_zt_adjusted, data['ZT_Adjusted'].min())
            max_zt_adjusted = max(max_zt_adjusted, data['ZT_Adjusted'].max())

        # Smooth the data with Gaussian filter (sigma=1)
        smoothed_data = {
            stage: gaussian_filter1d(data[stage], sigma=1)
            for stage in ['wake_percent', 'non_rem_percent', 'rem_percent']
        }

        # Identify gaps in the ZT sequence and insert NaN
        adjusted = [data['ZT_Adjusted'].iloc[0]]
        for i in range(1, len(data)):
            if data['ZT_Adjusted'].iloc[i] - data['ZT_Adjusted'].iloc[i - 1] > 1:
                # Insert NaN for the gap
                adjusted.append(np.nan)
            adjusted.append(data['ZT_Adjusted'].iloc[i])

        for stage in ['wake_percent', 'non_rem_percent', 'rem_percent']:
            values = smoothed_data[stage].tolist()
            adjusted_values = [values[0]]
            for i in range(1, len(values)):
                if data['ZT_Adjusted'].iloc[i] - data['ZT_Adjusted'].iloc[i - 1] > 1:
                    # Insert NaN for the gap
                    adjusted_values.append(np.nan)
                adjusted_values.append(values[i])

            # Plot the adjusted data
            ax1.plot(adjusted, adjusted_values, linestyle=line_styles[idx % len(line_styles)], 
                     color=colors[stage], linewidth=1, label=f"{stage_labels[stage]} ({subject})")

    # Handle case where no valid data exists
    if max_zt_adjusted == -float('inf') or min_zt_adjusted == float('inf'):
        print("No valid data found to plot.")
        return

    # Add a 12-hour cycle bar directly below the graph using rectangles
    total_cycles = int(np.ceil(max_zt_adjusted / 12))  # Number of full 12-hour cycles
    y_min, y_max = ax1.get_ylim()
    bar_height = (y_max - y_min) * 0.05  # Height of the bar as 5% of the y-axis range
    bar_y_start = y_min - bar_height  # Position the bar just below the visible range

    for i in range(total_cycles):
        start = i * 12
        end = (i + 1) * 12
        color = 'orange' if i % 2 == 0 else 'grey'

        # Add the rectangle for the cycle bar
        ax1.add_patch(Rectangle((start, bar_y_start), width=end - start, height=bar_height, color=color, alpha=0.6))

    # Adjust the y-axis limits to include the cycle bar
    ax1.set_ylim(bar_y_start, y_max)

    # Adjust the x-axis to cover the full range of ZT_Adjusted
    ax1.set_xlim(min_zt_adjusted, max_zt_adjusted)

    # Adjust the x-axis ticks and labels
    xtick_locations = np.arange(12, max_zt_adjusted + 1, 12)  # Every 12 hours
    xtick_labels = [f"{int(i)}" for i in xtick_locations]  # Label as ZT in cycles
    ax1.set_xticks(xtick_locations)
    ax1.set_xticklabels(xtick_labels, fontsize=18)
    ax1.tick_params(axis='y', labelsize=18)

    # Adding labels, legend, and title to the main plot
    ax1.set_xlabel('Zeitgeber Time', fontsize=18)
    ax1.set_ylabel('Stage Percent', fontsize=18)
    ax1.set_title('Vigilance Stage Across Zeitgeber Time', fontsize=24, pad=40)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.05, 1.40), ncol=2, fontsize=12)
    
    # Remove top and right spines for cleaner plot
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
   
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
plot_sleep_stages(['/Volumes/harris/volkan/hypnose/analysis/downsample_auto_score/sub-015_ses-02_recording-01_time-0-72h_sr-1hz_stitched_1hrbins_ZT.csv', 
                   '/Volumes/harris/volkan/hypnose/analysis/downsample_auto_score/sub-016_ses-03_recording-01_time-0-85h_sr-1hz_stitched_1hrbins_ZT.csv'], 
                  ['sub-015', 'sub-016'])
