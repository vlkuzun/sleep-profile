import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import glob

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

def analyze_sleep_cycles(file_path):
    """
    Analyze sleep cycles from a single CSV file.
    Returns DataFrame with start_time, end_time, and cycle_length (in minutes).
    """
    # Read CSV file and ensure timestamp is datetime type
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Find all REM stage end times
    rem_ends = []
    for i in range(len(df)-1):
        if df.iloc[i]['sleepStage'] == 3 and df.iloc[i+1]['sleepStage'] != 3:
            rem_ends.append(df.iloc[i]['Timestamp'])
    
    # Calculate cycle information
    cycles_data = []
    for i in range(len(rem_ends)-1):
        start_time = rem_ends[i]
        end_time = rem_ends[i+1]
        
        # Get all rows between these two REM end times
        mask = (df['Timestamp'] > start_time) & (df['Timestamp'] < end_time)
        between_stages = df[mask]
        
        # Check for Stage 1 periods longer than 2 minutes
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
        
        # If there's an ongoing Stage 1 at the end, check its duration
        if stage1_start is not None:
            stage1_duration = end_time - stage1_start
            if stage1_duration > timedelta(minutes=2):
                invalid_cycle = True
        
        # If cycle is valid, add it to the results
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

def plot_histogram(cycles_data, save_path=None):
    """
    Plot a histogram of cycle lengths and optionally save it as a high-resolution image.
    
    Args:
        cycles_data: DataFrame containing cycle length data
        save_path: Optional path where to save the figure (if None, just displays the plot)
    """
    plt.figure(figsize=(10, 6))
    plt.hist(cycles_data['cycle_length'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Cycle Length (minutes)', fontsize=20)
    plt.ylabel('Number of Cycles', fontsize=20)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        pdf_path = save_path[:-4] + '.pdf' if save_path.lower().endswith('.png') else f"{save_path}.pdf"
        plt.savefig(pdf_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to {save_path} and {pdf_path}")
    
    plt.show()

# Example usage:
folder_path = "/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis"
all_cycles = process_multiple_files(folder_path)

# Plot histogram
output_path = "/Volumes/harris/volkan/sleep-profile/plots/sleep_cycle/cycle_length_histogram_all_sub.png"
plot_histogram(all_cycles, save_path=output_path)

# Print average cycle length
average_cycle_length = all_cycles['cycle_length'].mean()
print(f"Average Cycle Length: {average_cycle_length:.2f} minutes")
