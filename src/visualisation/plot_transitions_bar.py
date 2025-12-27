import pandas as pd
import matplotlib.pyplot as plt
import os

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

def plot_sleep_transitions_multiple(files, output_dir=None, dpi=600):
    # Initialize a dictionary to count the transitions across all files
    transitions = {
        (1, 2): 0,  # Awake to NREM (1 -> 2)
        (2, 3): 0,  # NREM to REM (2 -> 3)
        (3, 1): 0,  # REM to Awake (3 -> 1)
        (2, 1): 0,  # NREM to Awake (2 -> 1)
        (3, 2): 0,  # REM to NREM (3 -> 2)
        (1, 3): 0   # Awake to REM (1 -> 3)
    }
    
    # Loop over the files and process each one
    for file_path in files:
        # Load the current CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Ensure that the sleepStage column exists in the current file
        if 'sleepStage' not in df.columns:
            raise ValueError(f"The file {file_path} must contain a 'sleepStage' column")
        
        # Loop through the dataframe and count the transitions
        for i in range(len(df) - 1):
            current_stage = df.iloc[i]['sleepStage']
            next_stage = df.iloc[i + 1]['sleepStage']
            
            if (current_stage, next_stage) in transitions:
                transitions[(current_stage, next_stage)] += 1
    
    # Calculate the total number of transitions
    total_transitions = sum(transitions.values())
    if total_transitions == 0:
        print("No transitions detected across the provided files.")
        return
    
    # Calculate the percentage of each transition
    transition_percentages = {
        key: (count / total_transitions) * 100 
        for key, count in transitions.items()
    }
    
    # Prepare data for plotting
    transition_labels = ['Wake-NREM', 'NREM-REM', 'REM-Wake', 
                         'NREM-Wake', 'REM-NREM', 'Wake-REM']
    transition_data = [transition_percentages[(1, 2)], transition_percentages[(2, 3)], 
                       transition_percentages[(3, 1)], transition_percentages[(2, 1)], 
                       transition_percentages[(3, 2)], transition_percentages[(1, 3)]]
    
    # Reverse the order of the labels for plotting
    transition_labels = transition_labels[::-1]
    transition_data = transition_data[::-1]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.barh(transition_labels, transition_data, color='#1f77b4', edgecolor='black', height=0.9)
    plt.xlabel('Percentage of Total Transitions', fontsize=20)
    plt.tick_params(axis='y', labelsize=18)
    plt.tick_params(axis='x', labelsize=18)
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, 50)
    
    plt.tight_layout()
    
    # Save the figure instead of showing it
    if output_dir is None:
        output_dir = os.getcwd()  # Use current working directory if none specified
    
    output_path = os.path.join(output_dir, 'transitions_bar_combined_sub.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    pdf_path = output_path[:-4] + '.pdf'
    plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {output_path} and {pdf_path} with DPI={dpi}")
    plt.close()

# Example usage:
files = ['/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_1Hz.csv', 
         '/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-010_ses-01_recording-01_time-0-69h_1Hz.csv', 
         '/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-011_ses-01_recording-01_time-0-72h_1Hz.csv',
         '/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-49h_1Hz_stitched.csv',
         '/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz.csv', 
         '/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz.csv']

# Specify output directory (optional) or use default current directory
output_dir = '/Volumes/harris/volkan/sleep-profile/plots/transitions'
plot_sleep_transitions_multiple(files, output_dir=output_dir, dpi=600)