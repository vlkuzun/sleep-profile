import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

# List of CSV file paths
csv_files = glob.glob('Z:/volkan/sleep_profile/downsample_auto_score/scoring_analysis/*.csv')

# Dictionary to store results
results = {}

# Initialize a list to hold the subject labels
subject_labels = []

# Define conditions
conditions = ['Wake Light', 'Wake Dark', 'NREM Light', 'NREM Dark', 'REM Light', 'REM Dark']

for file in csv_files:
    # Prompt for which subject this CSV file corresponds to
    subject_name = input(f"Enter the subject name for the file {file}: ")

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

    # Add the subject name to the labels for later plotting
    subject_labels.extend([subject_name] * len(bout_durations))

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

# Convert subject labels and data to a DataFrame for seaborn to handle the hue
plot_data = pd.DataFrame({
    'Condition': [cond for cond in conditions for _ in range(len(results))],
    'MeanBoutDuration': [item for sublist in stripplot_data.values() for item in sublist],
    'Subject': subject_labels_plot
})

# Ask the user to define colors for each subject
subject_palette = {}
for subject in plot_data['Subject'].unique():
    color = input(f"Enter a color for subject '{subject}': ")
    subject_palette[subject] = color

# Kruskal-Wallis test (non-parametric ANOVA alternative)
anova_data = plot_data[['Condition', 'MeanBoutDuration']]

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

# Create stripplot with 'hue' for different subjects, using the user-defined color map
sns.stripplot(x='Condition', y='MeanBoutDuration', data=plot_data, jitter=0.2, hue='Subject', palette=subject_palette, alpha=0.5, size=10, legend=False)

# Plot mean as a horizontal line for each condition
for i, cond in enumerate(conditions):
    condition_data = plot_data[plot_data['Condition'] == cond]
    condition_mean = condition_data['MeanBoutDuration'].mean()
    plt.hlines(condition_mean, i - 0.2, i + 0.2, color='black', linestyle='-', linewidth=2)

# Set initial offset above the y_max for the first comparison
y_max = plot_data['MeanBoutDuration'].max()
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
plt.xticks(rotation=45, fontsize=20)
plt.ylabel('Bout Duration (seconds)', fontsize=20)
plt.xlabel('')
plt.ylim(0, 650)  # Set y-axis range
plt.yticks(np.arange(0, 651, 200), fontsize=20)  # Reduced number of ticks, increased font

# Remove top and right spines
sns.despine(top=True, right=True)

plt.grid(axis='y', linestyle='--', alpha=0) # Hide grid
plt.tight_layout()
plt.savefig("Z:/volkan/sleep_profile/plots/bout_duration/bout_duration_grouped_comparison.png", dpi=600)
plt.show()
