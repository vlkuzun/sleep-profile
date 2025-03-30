import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set default font sizes globally before creating any plots
plt.rcParams.update({
    'font.size': 36,
    'axes.labelsize': 48,
    'axes.titlesize': 52,
    'xtick.labelsize': 44,
    'ytick.labelsize': 44,
    'legend.fontsize': 40,
})

# List of CSV file paths
csv_files = glob.glob('Z:/volkan/sleep_profile/downsample_auto_score/scoring_analysis/*.csv')

# List to store all bout durations from all files
all_bout_durations = []

for file in csv_files:
    # Load the CSV file
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
    
    # Add a column for the filename to distinguish data points from different files
    subject_name = file.split('/')[-1].split('.')[0]  # Extract subject name from filename (assuming subject name is in the file name)
    bout_durations['Subject'] = subject_name
    
    # Append to the master list
    all_bout_durations.append(bout_durations)

# Concatenate all bout durations into a single DataFrame
all_bout_durations_df = pd.concat(all_bout_durations, ignore_index=True)

# Assign unique colors to each subject
unique_subjects = all_bout_durations_df['Subject'].unique()
subject_palette = sns.color_palette("husl", len(unique_subjects))
subject_color_map = {subject: color for subject, color in zip(unique_subjects, subject_palette)}

# Prompt for colors if necessary
# For example, ask the user to define colors for each subject
for subject in unique_subjects:
    color = input(f"Enter a color for subject '{subject}': ")
    if color:
        subject_color_map[subject] = color

# Create separate plots for Wake, NREM, and REM
for sleep_stage in ['Wake', 'NREM', 'REM']:
    # Filter the data for the current sleep stage
    stage_data = all_bout_durations_df[all_bout_durations_df['sleepStage'] == sleep_stage]
    
    # Create figure with smaller size
    fig, ax = plt.subplots(figsize=(16, 10))  # was (20, 12)
    
    sns.stripplot(
        data=stage_data,
        x="timePeriod",
        y="boutDuration",
        hue="Subject",
        jitter=0.25,
        dodge=True,
        palette=subject_color_map,
        alpha=0.7,
        size=8,
        legend=False,
        ax=ax
    )
    
    # Get y-limits from data and set ticks
    _, ymax = ax.get_ylim()
    y_ticks = {
        'Wake': 15000,
        'NREM': 900,
        'REM': 300
    }
    tick_max = y_ticks[sleep_stage]
    yticks = np.linspace(0, tick_max, 4)
    ax.set_yticks(yticks)
    
    # Customize the plot with explicit font sizes
    ax.set_ylabel('Bout Duration (seconds)', fontsize=48)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=44)
    ax.tick_params(axis='y', labelsize=44)
    ax.set_title(f'{sleep_stage}', fontsize=52, pad=40)
    ax.grid(axis='y', linestyle='--', alpha=0)

    # Remove top and right spines
    sns.despine(top=True, right=True)

    # Save high-resolution figure
    plt.tight_layout()
    plt.savefig(f'Z:/volkan/sleep_profile/plots/bout_duration/bout_duration_individual_{sleep_stage}.png', 
                dpi=600,
                bbox_inches='tight',
                facecolor='white')
    plt.close()  # Close the figure to free memory
