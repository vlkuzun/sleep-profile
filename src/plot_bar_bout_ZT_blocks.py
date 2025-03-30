import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Set default font sizes globally before creating any plots
plt.rcParams.update({
    'font.size': 36,
    'axes.labelsize': 48,
    'axes.titlesize': 52,
    'xtick.labelsize': 44,
    'ytick.labelsize': 44,
    'legend.fontsize': 40,
})

def calculate_bout_durations_from_csv(file_path):
    """
    Calculate bout durations for each sleep stage based on continuous values from a CSV file.
    """
    def calculate_zt(timestamp):
        """
        Calculate the ZT (Zeitgeber Time) based on the timestamp.
        """
        base_time = datetime.strptime("09:00:00", "%H:%M:%S")
        # Remove the microseconds part by splitting at the dot if present
        time_str = timestamp.split()[1].split('.')[0]
        current_time = datetime.strptime(time_str, "%H:%M:%S")
        delta = current_time - base_time

        if delta.days < 0:  # Handle times past midnight
            delta += timedelta(days=1)

        zt = (delta.total_seconds() / 3600) % 24
        return zt

    # Read the data from the CSV
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        data = [(row['Timestamp'], int(row['sleepStage'])) for row in reader]

    if not data:
        return []

    # Initialize variables
    bouts = []
    start_timestamp = data[0][0]  # Start timestamp of the current bout
    current_stage = data[0][1]   # Current sleep stage
    bout_length = 1              # Length of the current bout

    # Iterate through the data starting from the second element
    for i in range(1, len(data)):
        timestamp, sleep_stage = data[i]

        if sleep_stage == current_stage:
            # Increment bout length if the stage is continuous
            bout_length += 1
        else:
            # Append the bout as a dictionary
            bouts.append({
                'Timestamp': start_timestamp,
                'Duration': bout_length,
                'sleepStage': current_stage,
                'ZT': calculate_zt(start_timestamp)
            })

            # Reset for the next bout
            start_timestamp = timestamp
            current_stage = sleep_stage
            bout_length = 1

    # Add the final bout to the result
    bouts.append({
        'Timestamp': start_timestamp,
        'Duration': bout_length,
        'sleepStage': current_stage,
        'ZT': calculate_zt(start_timestamp)
    })

    return bouts

def analyze_relationship_with_bar_charts_and_repeated_measures_anova(bout_data, subject_labels):
    """
    Create bar charts for ZT (divided into 3-hour blocks) and bout durations for each sleep stage across subjects.
    Perform repeated measures ANOVA and post-hoc tests for each sleep stage.
    """
    # Mapping sleep stages to names
    sleep_stage_map = {1: 'Wake', 2: 'NREM', 3: 'REM'}  # Changed 'Awake' to 'Wake'
    
    sleep_stages = sorted(set(bout['sleepStage'] for subject_data in bout_data for bout in subject_data))

    # Group ZT into 3-hour blocks
    def zt_to_block(zt):
        return int(zt // 3) * 3

    for stage in sleep_stages:
        # Map sleep stage to name
        stage_name = sleep_stage_map.get(stage, f"Stage {stage}")
        
        zt_blocks = []
        durations = []
        subjects = []

        # Collect data for plotting and analysis
        for idx, subject_data in enumerate(bout_data):
            stage_data = [bout for bout in subject_data if bout['sleepStage'] == stage]
            zt_blocks.extend(zt_to_block(bout['ZT']) for bout in stage_data)
            durations.extend(bout['Duration'] for bout in stage_data)
            subjects.extend([subject_labels[idx]] * len(stage_data))

        # Create a DataFrame for analysis
        plot_data = pd.DataFrame({
            'Subject': subjects,
            'ZT Block': zt_blocks,
            'Duration': durations
        })

        # Ensure all ZT blocks are represented
        all_blocks = range(0, 24, 3)
        grouped_data = plot_data.groupby(['Subject', 'ZT Block'])['Duration'].mean().unstack(fill_value=0)

        # Bar chart for mean duration per ZT block
        summary_data = plot_data.groupby('ZT Block')['Duration'].agg(['mean', sem]).reindex(all_blocks, fill_value=0)
        summary_data['ZT Block'] = summary_data.index

        plt.figure(figsize=(16, 10))
        colors = ['orange'] * 4 + ['grey'] * 4
        plt.bar(summary_data['ZT Block'], summary_data['mean'], color=colors, width=2.5, align='center', label='Mean Duration')

        # Plot mean for each subject (use a line or different markers)
        subject_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # New color order
        for idx, subject_data in enumerate(bout_data):
            subject_stage_data = [bout for bout in subject_data if bout['sleepStage'] == stage]
            subject_zt_blocks = [zt_to_block(bout['ZT']) for bout in subject_stage_data]
            subject_durations = [bout['Duration'] for bout in subject_stage_data]

            # Calculate the mean for each subject per ZT block
            subject_means = [np.mean([d for z, d in zip(subject_zt_blocks, subject_durations) if z == block]) for block in all_blocks]

            # Plot mean for each subject with specified colors
            plt.plot(all_blocks, subject_means, marker='o', linestyle='-', alpha=0.5, color=subject_colors[idx])

        # Title and labels
        plt.title(f'{stage_name}', fontsize=52, pad=40)
        plt.xlabel('ZT Block (hours)', fontsize=46)
        plt.ylabel('Bout Duration (seconds)', fontsize=46)

        plt.xticks(ticks=np.arange(0, 24, 3), labels=[f'{i}-{i+3}' for i in range(0, 24, 3)], fontsize=44, rotation=45)
        plt.tick_params(axis='y', labelsize=44)

        # Add specific y-ticks for NREM state
        if stage_name == 'NREM':
            plt.yticks(np.linspace(0, 210, 4))  # Creates 4 ticks from 0 to 150

        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Adjust bottom margin to prevent x-label trimming
        plt.subplots_adjust(bottom=0.2)
        
        # Show the plot without the legend
        plt.tight_layout()
        plt.savefig(f'Z:/volkan/sleep_profile/plots/bout_duration/bout_duration_across_ZT_{stage_name}.png', dpi=600, bbox_inches='tight')
        plt.show()

        # Repeated measures ANOVA
        melted_data = grouped_data.reset_index().melt(id_vars='Subject', var_name='ZT_Block', value_name='Duration')
        melted_data['ZT_Block'] = melted_data['ZT_Block'].astype(str)  # Ensure ZT Block is categorical
        anova_model = AnovaRM(melted_data, depvar='Duration', subject='Subject', within=['ZT_Block'])
        anova_result = anova_model.fit()
        print(f"{stage_name} Repeated Measures ANOVA:")
        print(anova_result)

        # Post-hoc pairwise comparisons
        mc = MultiComparison(melted_data['Duration'], melted_data['ZT_Block'])
        tukey_result = mc.tukeyhsd()
        print(f"Tukey HSD Post-hoc Test ({stage_name}):")
        print(tukey_result.summary())


# Example usage:
file_paths = [
    "Z:/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_1Hz.csv",
    "Z:/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-010_ses-01_recording-01_time-0-69h_1Hz.csv",
    "Z:/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-011_ses-01_recording-01_time-0-72h_1Hz.csv",
    "Z:/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-49h_1Hz_stitched.csv",
    "Z:/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz.csv",
    "Z:/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz.csv"    
]  # Add more file paths as needed
subject_labels = ["007", "010", "011", "015", "016", "017"]  # Replace with subject identifiers

all_bout_data = [calculate_bout_durations_from_csv(file_path) for file_path in file_paths]
analyze_relationship_with_bar_charts_and_repeated_measures_anova(all_bout_data, subject_labels)
