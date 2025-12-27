import csv
from datetime import datetime, timedelta
from scipy.stats import sem
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# Shared plotting style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


def _subject_sort_key(label):
    match = re.search(r"\d+", str(label))
    return int(match.group()) if match else float('inf')


def _normalize_subject_label(label):
    match = re.search(r"\d+", str(label))
    if match:
        return f"sub-{match.group().zfill(3)}"
    return str(label)

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
    sleep_stage_map = {1: 'Wake', 2: 'NREM', 3: 'REM'}

    normalized_subjects = [_normalize_subject_label(label) for label in subject_labels]
    ordered_subjects = sorted(normalized_subjects, key=_subject_sort_key)
    cmap = plt.get_cmap('tab10')
    subject_palette = {subject: cmap(idx % cmap.N) for idx, subject in enumerate(ordered_subjects)}

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
            subjects.extend([normalized_subjects[idx]] * len(stage_data))

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

        fig, ax = plt.subplots(figsize=(10, 6))
        light_color = '#FFD1A1'
        dark_color = '#C0C0C0'
        colors = [light_color if block < 12 else dark_color for block in all_blocks]
        ax.bar(summary_data['ZT Block'], summary_data['mean'], color=colors, width=2.5, align='center')

        # Plot mean for each subject (use a line or different markers)
        for idx, subject_data in enumerate(bout_data):
            subject_stage_data = [bout for bout in subject_data if bout['sleepStage'] == stage]
            subject_zt_blocks = [zt_to_block(bout['ZT']) for bout in subject_stage_data]
            subject_durations = [bout['Duration'] for bout in subject_stage_data]
            subject_label = normalized_subjects[idx]

            # Calculate the mean for each subject per ZT block
            subject_means = [np.mean([d for z, d in zip(subject_zt_blocks, subject_durations) if z == block]) for block in all_blocks]

            # Plot mean for each subject with specified colors
            ax.plot(all_blocks, subject_means, marker='o', linestyle='-', alpha=0.5, color=subject_palette[subject_label], linewidth=1.2, markersize=4)

        # Title and labels
        ax.set_title(stage_name, fontsize=22, pad=20)
        ax.set_xlabel('Zeitgeber time (ZT)', fontsize=20)
        ax.set_ylabel('Bout Duration (seconds)', fontsize=20)
        xtick_positions = np.arange(0, 24, 3)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f'{i}-{i+3}' for i in xtick_positions], rotation=15)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        if stage_name == 'NREM':
            ax.set_yticks(np.linspace(0, 210, 4))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        plt.tight_layout()
        output_png = f'/Volumes/harris/volkan/sleep-profile/plots/bout_duration/bout_duration_across_ZT_{stage_name}.png'
        fig.savefig(output_png, dpi=600, bbox_inches='tight')
        output_pdf = output_png[:-4] + '.pdf'
        fig.savefig(output_pdf, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close(fig)

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
    "/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_1Hz.csv",
    "/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-010_ses-01_recording-01_time-0-69h_1Hz.csv",
    "/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-011_ses-01_recording-01_time-0-72h_1Hz.csv",
    "/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-49h_1Hz_stitched.csv",
    "/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz.csv",
    "/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz.csv"    
]  # Add more file paths as needed
subject_labels = ["sub-007", "sub-010", "sub-011", "sub-015", "sub-016", "sub-017"]

all_bout_data = [calculate_bout_durations_from_csv(file_path) for file_path in file_paths]
analyze_relationship_with_bar_charts_and_repeated_measures_anova(all_bout_data, subject_labels)
