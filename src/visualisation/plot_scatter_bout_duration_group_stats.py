import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

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

# Dictionary to store results
results = {}

# Define conditions
conditions = ['Wake Light', 'Wake Dark', 'NREM Light', 'NREM Dark', 'REM Light', 'REM Dark']

for file in csv_files:
    # Determine subject name from file path (e.g., sub-007)
    subject_match = re.search(r"sub-(\d+)", file, re.IGNORECASE)
    if subject_match:
        subject_id = subject_match.group(1)
        subject_name = f"sub-{subject_id}"
    else:
        subject_name = Path(file).stem

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
    bout_durations = df.groupby(['boutId', 'sleepStage']).size().reset_index(name='boutDuration')
    
    # Map sleep stages to their corresponding names
    sleep_stage_map = {1: 'Wake', 2: 'NREM', 3: 'REM'}
    bout_durations['sleepStage'] = bout_durations['sleepStage'].map(sleep_stage_map)
    
    # Determine the time period for each bout based on the majority time
    bout_time_periods = df.groupby('boutId')['timePeriod'].apply(lambda x: x.value_counts().index[0]).reset_index()
    
    # Merge the bout durations with the time periods
    bout_durations = pd.merge(bout_durations, bout_time_periods, on='boutId')
    
    # Group by timePeriod and sleepStage, then calculate the mean bout duration
    light_dark_avg_duration_stages = bout_durations.groupby(['timePeriod', 'sleepStage'])['boutDuration'].mean()
    
    # Store results in dictionary, using subject name as the key
    results[subject_name] = {
        'bout_durations': bout_durations,
        'light_dark_avg_duration_stages': light_dark_avg_duration_stages,
    }

# Prepare data for plotting
stripplot_data = {cond: [] for cond in conditions}
subject_labels_plot = []  # List for subject labels corresponding to the plot

# Collect data for each condition
for time_period, sleep_stage in [('Light', 'Wake'), ('Dark', 'Wake'), 
                                 ('Light', 'NREM'), ('Dark', 'NREM'), 
                                 ('Light', 'REM'), ('Dark', 'REM')]:
    condition = f'{sleep_stage} {time_period}'
    
    # Collect the individual means for the condition across all subjects
    for subject_name, result in results.items():
        try:
            mean = result['light_dark_avg_duration_stages'].loc[(time_period, sleep_stage)]
            stripplot_data[condition].append(mean)
            subject_labels_plot.append(subject_name)
        except KeyError:
            stripplot_data[condition].append(np.nan)
            subject_labels_plot.append(subject_name)

# Convert subject labels and data to a DataFrame for plotting
plot_data = pd.DataFrame({
    'Condition': [cond for cond in conditions for _ in range(len(results))],
    'MeanBoutDuration': [item for sublist in stripplot_data.values() for item in sublist],
    'Subject': subject_labels_plot
})

# Prepare consistent subject colors matching other figures
def _subject_sort_key(subject_label):
    match = re.search(r"\d+", str(subject_label))
    return int(match.group()) if match else float('inf')

subjects = sorted(plot_data['Subject'].dropna().unique(), key=_subject_sort_key)
cmap = plt.get_cmap('tab10')
subject_palette = {subj: cmap(idx % cmap.N) for idx, subj in enumerate(subjects)}

# Remove rows without data for statistical tests and plotting
plot_data_clean = plot_data.dropna(subset=['MeanBoutDuration'])

# Kruskal-Wallis test (non-parametric ANOVA alternative)
anova_data = plot_data_clean[['Condition', 'MeanBoutDuration']]

# Kruskal-Wallis test for each condition
grouped_data = [anova_data[anova_data['Condition'] == cond]['MeanBoutDuration'] for cond in conditions]
kruskal_result = kruskal(*grouped_data)

# Print Kruskal-Wallis result
print("Kruskal-Wallis Test Results:")
print(f"Test statistic: {kruskal_result.statistic}")
print(f"P-value: {kruskal_result.pvalue}")

# Initialize significant_comparisons as empty list
significant_comparisons = []

# If Kruskal-Wallis test is significant, perform pairwise Dunn's test
if kruskal_result.pvalue < 0.05:
    print("\nPost-hoc Pairwise Comparison using Dunn's Test:")
    dunn_result = posthoc_dunn(anova_data, val_col='MeanBoutDuration', group_col='Condition', p_adjust='bonferroni')
    print(dunn_result)

    # Get the significant comparisons from Dunn's test (p < 0.05)
    significant_comparisons = dunn_result[dunn_result < 0.05].stack().index.tolist()

# Plot
plt.figure(figsize=(10, 6))

condition_positions = {cond: idx for idx, cond in enumerate(conditions)}
rng = np.random.default_rng(seed=42)

# Plot individual subject data with jitter per condition
for subject in plot_data_clean['Subject'].unique():
    subject_data = plot_data_clean[plot_data_clean['Subject'] == subject]
    x_vals = [condition_positions[cond] + rng.uniform(-0.15, 0.15) for cond in subject_data['Condition']]
    y_vals = subject_data['MeanBoutDuration'].values
    plt.scatter(x_vals, y_vals, color=subject_palette.get(subject, '#000000'), alpha=0.6, s=80)

# Plot mean as a horizontal line for each condition
for i, cond in enumerate(conditions):
    condition_data = plot_data_clean[plot_data_clean['Condition'] == cond]
    condition_mean = condition_data['MeanBoutDuration'].mean()
    plt.hlines(condition_mean, i - 0.2, i + 0.2, color='black', linestyle='-', linewidth=2)

# Set initial offset above the y_max for the first comparison
y_max = plot_data_clean['MeanBoutDuration'].max()
comparison_offset = y_max + 30  # Space above the maximum value of the y-axis for the first comparison

# Track comparisons to avoid duplicating
shown_comparisons = set()

# Add asterisks for significant post-hoc comparisons
for comparison in significant_comparisons:
    cond1, cond2 = comparison
    # Ensure comparisons are ordered to avoid duplicates (e.g., 'A vs B' and 'B vs A' should not both be shown)
    comparison_pair = tuple(sorted([cond1, cond2]))

    # Check if the comparison has already been shown
    if comparison_pair not in shown_comparisons:
        idx1 = conditions.index(cond1)
        idx2 = conditions.index(cond2)
        
        # Plot a line between the two conditions
        plt.plot([idx1, idx2], [comparison_offset, comparison_offset], color='black', lw=1.5)
        plt.text((idx1 + idx2) / 2, comparison_offset + 0.05, "*", ha='center', fontsize=20, color='black')

        # Add the comparison to the set to ensure it's not repeated
        shown_comparisons.add(comparison_pair)

        # Increase the offset for the next comparison to move it higher
        comparison_offset += 40  # Increase the gap for the next comparison

# Customize plot
plt.xticks(range(len(conditions)), conditions, rotation=45, fontsize=20)
plt.ylabel('Bout Duration (seconds)', fontsize=20)
plt.xlabel('')
plt.ylim(0, 650)  # Set y-axis range
plt.yticks(np.arange(0, 651, 200), fontsize=20)  # Reduced number of ticks, increased font

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.grid(axis='y', linestyle='--', alpha=0) # Hide grid
plt.tight_layout()
output_png = "/Volumes/harris/volkan/sleep-profile/plots/bout_duration/bout_duration_grouped_comparison.png"
plt.savefig(output_png, dpi=600)
output_pdf = output_png[:-4] + '.pdf'
plt.savefig(output_pdf, dpi=600)
plt.show()
