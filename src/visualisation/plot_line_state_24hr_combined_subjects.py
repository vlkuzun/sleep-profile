import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_combined_sleep_data(input_file, output_file):
    # Load the combined CSV file with all subjects' data
    df = pd.read_csv(input_file)

    # Ensure data is sorted by ZT for proper plotting
    df = df.sort_values(by='ZT')

    # Calculate the overall mean and SEM across all subjects for each ZT
    mean_df = df.groupby('ZT').mean(numeric_only=True)[['wake_percent_mean', 'non_rem_percent_mean', 'rem_percent_mean']]
    sem_df = df.groupby('ZT').sem(numeric_only=True)[['wake_percent_mean', 'non_rem_percent_mean', 'rem_percent_mean']]

    # Set up Seaborn style without gridlines
    plt.figure(figsize=(14, 6))

    # Define colors for each sleep stage
    stage_colors = {'wake_percent_mean': 'red', 'non_rem_percent_mean': 'blue', 'rem_percent_mean': 'green'}
    sleep_stages = ['wake_percent_mean', 'non_rem_percent_mean', 'rem_percent_mean']
    stage_titles = {'wake_percent_mean': 'Wake', 'non_rem_percent_mean': 'NREM', 'rem_percent_mean': 'REM'}

    # Plot mean and SEM for each sleep stage with specified colors
    for stage in sleep_stages:
        # Plot mean line with its actual color
        sns.lineplot(
            x=mean_df.index, y=mean_df[stage],
            color=stage_colors[stage], linewidth=2.5, label=f'{stage_titles[stage]}'
        )

        # Plot SEM as shaded area
        plt.fill_between(
            mean_df.index,
            mean_df[stage] - sem_df[stage],
            mean_df[stage] + sem_df[stage],
            color=stage_colors[stage], alpha=0.3
        )

    # Add phase indicator bar at the bottom of the plot
    ax = plt.gca()
    # Add lighter orange bar for ZT 0-1
    ax.add_patch(Rectangle((0, -5), 1, 5, color='#FFD1A1', alpha=0.8, lw=0))
    # Add regular orange bar for ZT 1-12
    ax.add_patch(Rectangle((1, -5), 11, 5, color='orange', alpha=0.8, lw=0))
    # Add brighter gray bar for ZT 12-13
    ax.add_patch(Rectangle((12, -5), 1, 5, color='#C0C0C0', alpha=0.8, lw=0))
    # Add regular gray bar for ZT 13-23
    ax.add_patch(Rectangle((13, -5), 10, 5, color='gray', alpha=0.8, lw=0))

    # Customize x-axis to show all ZT points and ensure whole integers
    xticks_range = range(int(mean_df.index.min()), int(mean_df.index.max()) + 1)
    plt.xticks(ticks=xticks_range, fontsize=22)
    ax.set_xlim(left=min(xticks_range), right=max(xticks_range))

    # Set y-axis to start from 0 and create y-ticks every 20 units
    plt.ylim(bottom=-5)
    plt.yticks(ticks=range(0, 101, 20), fontsize=22)  # Changed from 10 to 20

    # Set titles and labels
    plt.xlabel("Zeitgeber Time (ZT)", fontsize=24, labelpad=20)
    plt.ylabel("Percentage", fontsize=24)

    # Adjust legend to remove the title and make the box smaller and text clearer
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 1.1), title=None, fontsize=18, frameon=True, framealpha=0.7, markerscale=1.2)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tight layout for the plot
    plt.tight_layout()

    # Save the plot to the specified output location
    #plt.show() # Testing
    plt.savefig(output_file, dpi=600)
    plt.close()  # Close the figure after saving to free memory
    print(f"Plot saved to: {output_file}")

# Get input file path and output file path
input_file = input("Enter the path of the combined CSV file: ")
output_file = input("Enter the output file path for the plot (e.g., path/to/plot.png): ")
plot_combined_sleep_data(input_file, output_file)
