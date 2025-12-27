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

    # Function to calculate bout lengths regardless of sleep stage
    def calculate_bouts(df):
        # Identify bouts regardless of sleep stage
        df['bout'] = (df['sleepStage'] != df['sleepStage'].shift()).cumsum()
        
        # Group by 'bout' and 'phase' to calculate bout lengths
        bout_lengths = df.groupby(['bout', 'phase']).size().reset_index(name='length')
        return bout_lengths

    # Calculate bout lengths across all sleep stages
    bout_lengths_df = calculate_bouts(df)

    # Separate bout lengths by light and dark phases
    light_bouts = bout_lengths_df[bout_lengths_df['phase'] == 'light']['length']
    dark_bouts = bout_lengths_df[bout_lengths_df['phase'] == 'dark']['length']

    # Run t-test to compare bout lengths between light and dark phases
    t_stat, p_val = ttest_ind(light_bouts, dark_bouts, equal_var=False)

    # Prepare data for plotting (mean and SEM for each phase)
    bout_summary = bout_lengths_df.groupby('phase')['length'].agg(['mean', 'sem']).reset_index()

    # Plot results
    plt.figure(figsize=(8, 6))
    # Passing yerr as a list of SEM values to match each bar
    sns.barplot(data=bout_summary, x='phase', y='mean', yerr=bout_summary['sem'].values, capsize=0.2)
    plt.xlabel("Phase")
    plt.ylabel("Average Bout Length (minutes)")
    plt.title("Comparison of Average Bout Length in Light vs Dark Phases")

    # Add statistical annotation
    y_max = bout_summary['mean'].max() + bout_summary['sem'].max() + 0.5
    plt.text(0.5, y_max, f'p = {p_val:.3f}', ha='center', fontsize=12, fontweight='bold')

    # Save plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close() 
    print(f"Plot saved to: {output_file}")

# Get input file path and output file path
input_file = input("Enter the path of the combined CSV file: ")
output_file = input("Enter the output file path for the plot (e.g., path/to/plot.png): ")
analyze_and_plot_bout_lengths(input_file, output_file)
