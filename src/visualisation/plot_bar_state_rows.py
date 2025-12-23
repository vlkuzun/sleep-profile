import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def _ensure_repo_root_on_path():
    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "src" / "stage_colors.py").exists():
            repo_root = parent
            break
    else:
        repo_root = repo_root.parent

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.append(repo_root_str)


_ensure_repo_root_on_path()

from src.stage_colors import get_stage_color
# File: /Users/Volkan/Repos/sleep-profile/src/plot_bar_state_rows.py

# Default output location for saved plots
DEFAULT_SAVE_PATH = '/Volumes/harris/volkan/sleep-profile/plots/state_rows/sleep_stage_rows.png'

# Set global style for publication consistency
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

def plot_sleep_stages(csv_file, start_time, end_time, save_path=None):
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
    colors = {
        1: get_stage_color('Wake'),
        1.5: '#D55E00',  # Microarousals keep a distinct color
        2: get_stage_color('NREM'),
        3: get_stage_color('REM'),
    }
    y_pos = 0
    total_length = 0
    for _, row in stage_info.iterrows():
        color = colors.get(row['plot_stage'], '#999999')
            
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

    # Save figure if a path is provided (PNG + PDF).
    # Default to global path when none is passed in.
    output_path = save_path or DEFAULT_SAVE_PATH
    if output_path:
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        if output_path.endswith('.png'):
            pdf_path = output_path.replace('.png', '.pdf')
        else:
            pdf_path = f"{output_path}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.show()

# Example usage
start_time = '2024-11-30 11:30:00'
end_time = '2024-11-30 13:00:00'
plot_sleep_stages(
    '/Volumes/harris/volkan/sleep-profile/downsample_auto_score/scoring_analysis_consolidated/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz_consolidated.csv',
    start_time,
    end_time,
    save_path='/Volumes/harris/volkan/sleep-profile/plots/state_bars/state_bars_sub-016_ZT2.5-4.png'
)