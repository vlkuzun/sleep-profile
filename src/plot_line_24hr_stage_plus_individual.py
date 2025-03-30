import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_combined_sleep_data(input_file, output_file):
    # Load the combined CSV file with all subjects' data
    df = pd.read_csv(input_file)

    # Ensure data is sorted by ZT for proper plotting
    df = df.sort_values(by='ZT')

    # Calculate the overall mean and SEM across all subjects for each ZT
    mean_df = df.groupby('ZT').mean(numeric_only=True)[['wake_percent_mean', 'non_rem_percent_mean', 'rem_percent_mean']]
    sem_df = df.groupby('ZT').sem(numeric_only=True)[['wake_percent_mean', 'non_rem_percent_mean', 'rem_percent_mean']]

    # Set up Seaborn style without gridlines
    sns.set(style="white")
    plt.figure(figsize=(12, 8))

    # Define colors for each sleep stage
    stage_colors = {'wake_percent_mean': 'red', 'non_rem_percent_mean': 'blue', 'rem_percent_mean': 'green'}
    sleep_stages = ['wake_percent_mean', 'non_rem_percent_mean', 'rem_percent_mean']
    stage_titles = {'wake_percent_mean': 'Wake', 'non_rem_percent_mean': 'Non-REM', 'rem_percent_mean': 'REM'}

    # Define dashed line styles for individual subjects
    line_styles = ['-', '--', '-.', ':', (0, (5, 5)), (0, (3, 5, 1, 5))]  # Extended with custom dash styles

    # Shade background for ZT 12-23 (representing dark phase)
    plt.axvspan(12, 23, color='gray', alpha=0.3)

    # Plot each subject's individual data with matching colors for each stage and dashed line style
    subjects = df['subject'].unique()
    for idx, subject in enumerate(subjects):
        subject_data = df[df['subject'] == subject]
        line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles
        for stage in sleep_stages:
            sns.lineplot(
                data=subject_data, x='ZT', y=stage,
                color=stage_colors[stage], linestyle=line_style, linewidth=1,
                label=f'{subject}' if stage == 'wake_percent_mean' else ""  # Label once per subject
            )

    # Plot mean and SEM for each sleep stage with specified colors
    for stage in sleep_stages:
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

    # Customize x-axis to show all ZT points
    plt.xticks(ticks=mean_df.index, labels=mean_df.index)
    
    # Set titles and labels
    plt.title("Average vigilance stage (with standard error)")
    plt.xlabel("Zeitgeber time (ZT)")
    plt.ylabel("Percentage (%)")

    # Adjust legend to make the box smaller and text clearer
    plt.legend(loc='upper left', title="Legend", fontsize=10, frameon=True, framealpha=0.7, markerscale=1.2)

    # Tight layout for the plot
    plt.tight_layout()

    # Save the plot to the specified output location
    plt.show() #Testing
    #plt.savefig(output_file)
    plt.close()  # Close the figure after saving to free memory
    print(f"Plot saved to: {output_file}")

# Get input file path and output file path
input_file = input("Enter the path of the combined CSV file: ")
output_file = input("Enter the output file path for the plot (e.g., path/to/plot.png): ")
plot_combined_sleep_data(input_file, output_file)
