import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import glob
import numpy as np
from scipy.stats import pearsonr

def analyze_sleep_cycles(file_path):
    """
    Analyze sleep cycles from a single CSV file.
    Returns DataFrame with start_time, end_time, and cycle_length (in minutes).
    """
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    rem_ends = []
    for i in range(len(df)-1):
        if df.iloc[i]['sleepStage'] == 3 and df.iloc[i+1]['sleepStage'] != 3:
            rem_ends.append(df.iloc[i]['Timestamp'])
    
    cycles_data = []
    for i in range(len(rem_ends)-1):
        start_time = rem_ends[i]
        end_time = rem_ends[i+1]
        
        mask = (df['Timestamp'] > start_time) & (df['Timestamp'] < end_time)
        between_stages = df[mask]
        
        invalid_cycle = False
        stage1_start = None
        
        for idx, row in between_stages.iterrows():
            if row['sleepStage'] == 1:
                if stage1_start is None:
                    stage1_start = row['Timestamp']
            else:
                if stage1_start is not None:
                    stage1_duration = row['Timestamp'] - stage1_start
                    if stage1_duration > timedelta(minutes=2):
                        invalid_cycle = True
                        break
                    stage1_start = None
        
        if stage1_start is not None:
            stage1_duration = end_time - stage1_start
            if stage1_duration > timedelta(minutes=2):
                invalid_cycle = True
        
        if not invalid_cycle:
            cycles_data.append({
                'start_time': start_time,
                'end_time': end_time,
                'cycle_length': (end_time - start_time).total_seconds() / 60
            })
    
    return pd.DataFrame(cycles_data)

def process_multiple_files(folder_path):
    """
    Process multiple CSV files in the specified folder.
    Returns a combined DataFrame of cycle lengths.
    """
    all_files = glob.glob(f"{folder_path}/*.csv")
    combined_data = pd.DataFrame()
    
    for file in all_files:
        df = analyze_sleep_cycles(file)
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    
    return combined_data

def convert_to_zt(start_time):
    """
    Convert a timestamp to Zeitgeber Time (ZT) with 09:00:00 as ZT 0.
    """
    zt_zero = start_time.replace(hour=9, minute=0, second=0, microsecond=0)
    zt_offset = (start_time - zt_zero).total_seconds() / 3600
    return zt_offset % 24

def add_zt_column(cycles_data):
    """
    Add ZT (Zeitgeber Time) column to the cycles data DataFrame.
    """
    cycles_data['ZT'] = cycles_data['start_time'].apply(convert_to_zt)
    return cycles_data

def plot_cycle_length_vs_zt(cycles_data):
    """
    Plot a scatter plot of cycle lengths vs. ZT with regression line using sns.regplot.
    """
    # Add ZT column
    cycles_data = add_zt_column(cycles_data)
    
    # Calculate the correlation between ZT and cycle length
    correlation, p_value = pearsonr(cycles_data['ZT'], cycles_data['cycle_length'])
    
    # Regression plot with seaborn
    plt.figure(figsize=(12, 6))
    sns.regplot(x='ZT', y='cycle_length', data=cycles_data, scatter_kws={'alpha':0.7, 'color':'blue'}, line_kws={'color':'red', 'lw':2})

    plt.title('Sleep Cycle Length vs. Zeitgeber Time (ZT)')
    plt.xlabel('Zeitgeber Time (ZT)')
    plt.ylabel('Cycle Length (minutes)')
    plt.xticks(np.arange(0, 25, 2))  # Tick marks every 2 ZT hours
    plt.grid(alpha=0.3)

    # Display correlation coefficient on the plot
    plt.text(1, max(cycles_data['cycle_length']), f'Correlation: {correlation:.2f}', fontsize=12, color='black')

    plt.tight_layout()
    plt.show()
    
    # Print the correlation result
    print(f"Correlation between ZT and Cycle Length: {correlation:.2f}")
    print(f"P-value: {p_value:.4f}")

# Example usage:
folder_path = "/Volumes/harris/volkan/sleep_profile/downsample_auto_score/bout_duration"
all_cycles = process_multiple_files(folder_path)
plot_cycle_length_vs_zt(all_cycles)
