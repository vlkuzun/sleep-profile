import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def analyze_and_plot_bout_lengths(input_file, output_file):
    # Load data
    df = pd.read_csv(input_file)

    # Assume the CSV has columns: 'Timestamp' and 'sleepStage' (1, 2, or 3)
    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Extract time from Timestamp to classify light and dark phases
    df['time'] = df['Timestamp'].dt.time

    # Define light and dark phases
    light_start, light_end = pd.to_datetime("09:00").time(), pd.to_datetime("21:00").time()
    
    # Helper function to classify time as light or dark phase
    def phase(time):
        return 'light' if light_start <= time < light_end else 'dark'

    # Apply phase classification
    df['phase'] = df['time'].apply(phase)

    # Function to calculate bout lengths
    def calculate_bouts(df, stage):
        # Select data for specific stage
        df_stage = df[df['sleepStage'] == stage]
        df_stage['bout'] = (df_stage['sleepStage'] != df_stage['sleepStage'].shift()).cumsum()
        
        # Group by 'bout' to calculate lengths
        bout_lengths = df_stage.groupby(['bout', 'phase']).size().reset_index(name='length')
        return bout_lengths

    # Calculate bout lengths for each sleep stage
    bout_data = []
    for stage in [1, 2, 3]:
        bouts = calculate_bouts(df, stage)
        bouts['sleepStage'] = stage
        bout_data.append(bouts)
    bout_lengths_df = pd.concat(bout_data, ignore_index=True)

    # Run t-tests for each sleep stage
    t_test_results = {}
    for stage in [1, 2, 3]:
        light_bouts = bout_lengths_df[(bout_lengths_df['sleepStage'] == stage) & (bout_lengths_df['phase'] == 'light')]['length']
        dark_bouts = bout_lengths_df[(bout_lengths_df['sleepStage'] == stage) & (bout_lengths_df['phase'] == 'dark')]['length']
        t_stat, p_val = ttest_ind(light_bouts, dark_bouts, equal_var=False)
        t_test_results[stage] = p_val

    # Plot results
    plt.figure(figsize=(10, 6))
    sns.barplot(data=bout_lengths_df, x='sleepStage', y='length', hue='phase', ci='sd')
    plt.xlabel("Sleep Stage")
    plt.ylabel("Average Bout Length (minutes)")
    plt.title("Comparison of Bout Lengths in Light vs Dark Phases")

    # Add statistical annotations
    for stage in [1, 2, 3]:
        p_val = t_test_results[stage]
        y_max = bout_lengths_df[bout_lengths_df['sleepStage'] == stage]['length'].max()
        y = y_max + 0.5
        plt.text(stage - 1, y, f'p = {p_val:.3f}', ha='center')

    # Save plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to: {output_file}")

# Get input file path and output file path
input_file = input("Enter the path of the combined CSV file: ")
output_file = input("Enter the output file path for the plot (e.g., path/to/plot.png): ")
analyze_and_plot_bout_lengths(input_file, output_file)
