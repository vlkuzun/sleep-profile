import pandas as pd
import matplotlib.pyplot as plt

def plot_sleep_stages(csv_file, start_time, end_time):
    # Read and process CSV
    df = pd.read_csv(csv_file)
    if 'sleepStage' not in df.columns or 'Timestamp' not in df.columns:
        raise ValueError("CSV file must contain 'sleepStage' and 'Timestamp' columns")
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]
    
    # Process consecutive stages with previous stage information
    df['group'] = (df['sleepStage'] != df['sleepStage'].shift()).cumsum()
    df['prev_stage'] = df['sleepStage'].shift()
    
    # Group by consecutive stages and get lengths
    stage_info = df.groupby('group').agg({
        'sleepStage': 'first',
        'prev_stage': 'first',
        'Timestamp': 'size'
    }).rename(columns={'Timestamp': 'length'}).reset_index()
    
    # Set plot stage based on conditions
    stage_info['plot_stage'] = stage_info['sleepStage']
    stage_info.loc[
        (stage_info['sleepStage'] == 1) & 
        (stage_info['length'] < 40) & 
        (stage_info['prev_stage'] != 3), 
        'plot_stage'
    ] = 1.5
    
    # Map original stages to compressed y-axis positions
    stage_mapping = {1: 1.75, 1.5: 1.5, 2: 1.25, 3: 1.0}
    stage_info['plot_position'] = stage_info['plot_stage'].map(stage_mapping)
    
    # Plot with compressed y-axis spacing
    fig, ax = plt.subplots(figsize=(16, 1))
    y_pos = 0
    total_length = 0
    for _, row in stage_info.iterrows():
        if row['plot_stage'] == 1:
            color = 'red'
        elif row['plot_stage'] == 1.5:
            color = 'orange'
        elif row['plot_stage'] == 2:
            color = 'blue'
        else:  # stage 3
            color = 'green'
            
        ax.broken_barh([(y_pos, row['length'])], 
                      (row['plot_position'] - 0.1, 0.25),
                      facecolors=color)
        y_pos += row['length']
        total_length = y_pos  # Keep track of total length
    
    # Set x-axis limits to cover full width
    ax.set_xlim(0, total_length)
    
    # Remove decorative elements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
start_time = '2024-11-30 11:30:00'
end_time = '2024-11-30 13:00:00'
plot_sleep_stages('/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis_consolidated/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz_consolidated.csv', start_time, end_time)