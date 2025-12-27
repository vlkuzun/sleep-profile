import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from pathlib import Path

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

# List of CSV file paths
csv_files = glob.glob('/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/*.csv')

# List to store all bout durations from all files
all_bout_durations = []

for file in csv_files:
    df = pd.read_csv(file)
    
    # Convert Timestamp column to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Round the numbers in the sleepStage column to the nearest integer
    df['sleepStage'] = df['sleepStage'].round().astype(int)
    
    # Create a new column to track changes in sleep stage
    df['sleepStageChange'] = df['sleepStage'] != df['sleepStage'].shift()
    
    # Create a cumulative sum of changes to identify continuous instances
    df['boutId'] = df['sleepStageChange'].cumsum()
    
    # Determine the time period (light or dark) for each row
    def get_time_period(row):
        hour = row['Timestamp'].hour
        if 9 <= hour < 21:
            return 'Light'
        else:
            return 'Dark'
    
    df['timePeriod'] = df.apply(get_time_period, axis=1)
    
    # Group by boutId and sleepStage, then calculate the count of rows for each bout
    bout_durations = df.groupby(['boutId', 'sleepStage', 'timePeriod']).size().reset_index(name='boutDuration')
    
    # Map sleep stages to their corresponding names
    sleep_stage_map = {1: 'Wake', 2: 'NREM', 3: 'REM'}
    bout_durations['sleepStage'] = bout_durations['sleepStage'].map(sleep_stage_map)
    
    path_obj = Path(file)
    match = re.search(r"sub-(\d+)", path_obj.name, re.IGNORECASE)
    if match:
        subject_name = f"sub-{match.group(1)}"
    else:
        subject_name = path_obj.stem
    bout_durations['Subject'] = subject_name
    
    # Append to the master list
    all_bout_durations.append(bout_durations)

# Concatenate all bout durations into a single DataFrame
all_bout_durations_df = pd.concat(all_bout_durations, ignore_index=True)

def _subject_sort_key(label):
    match = re.search(r"\d+", str(label))
    return int(match.group()) if match else float('inf')

unique_subjects = sorted(all_bout_durations_df['Subject'].unique(), key=_subject_sort_key)
cmap = plt.get_cmap('tab10')
subject_color_map = {subject: cmap(idx % cmap.N) for idx, subject in enumerate(unique_subjects)}

# Create separate plots for Wake, NREM, and REM
for sleep_stage in ['Wake', 'NREM', 'REM']:
    # Filter the data for the current sleep stage
    stage_data = all_bout_durations_df[all_bout_durations_df['sleepStage'] == sleep_stage]
    
    fig, ax = plt.subplots(figsize=(10, 6))

    time_periods = ['Light', 'Dark']
    sns.stripplot(
        data=stage_data,
        x='timePeriod',
        y='boutDuration',
        hue='Subject',
        order=time_periods,
        dodge=True,
        palette=subject_color_map,
        jitter=0.25,
        alpha=0.7,
        size=6,
        ax=ax
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    
    # Get y-limits from data and set ticks
    y_limits = {
        'Wake': 16000,
        'NREM': 1000,
        'REM': 350
    }
    y_tick_max = {
        'Wake': 15000,
        'NREM': 900,
        'REM': 300
    }
    tick_max = y_tick_max[sleep_stage]
    yticks = np.linspace(0, tick_max, 4)
    ax.set_yticks(yticks)
    ax.set_ylim(0, y_limits[sleep_stage])
    
    # Customize the plot with explicit font sizes
    ax.set_xticks(range(len(time_periods)))
    ax.set_xticklabels(time_periods, fontsize=20)
    ax.set_ylabel('Bout Duration (seconds)', fontsize=20)
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=20)
    ax.set_title(f'{sleep_stage}', fontsize=22, pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save high-resolution figure
    plt.tight_layout()
    output_png = f'/Volumes/harris/volkan/sleep-profile/plots/bout_duration/bout_duration_individual_{sleep_stage}.png'
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    output_pdf = output_png[:-4] + '.pdf'
    plt.savefig(output_pdf, dpi=600, bbox_inches='tight')
    plt.close()
