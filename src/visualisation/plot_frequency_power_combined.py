import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path
from neurodsp.spectral import compute_spectrum


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

def plot_average_power_spectra(eeg_files, stage_files, channel, sampling_rate=512, output_file=None):
    """
    Plot the average power spectra for each sleep stage across multiple subjects.

    Parameters:
        eeg_files (list of str): Paths to the EEG signal pickle files.
        stage_files (list of str): Paths to the sleep stage scoring CSV files.
        channel (str): Name of the EEG channel to use.
        sampling_rate (int): Sampling rate of the EEG signal in Hz (default: 512).
        output_file (str): Path to save the output plot. If None, the plot is not saved.
    """
    if len(eeg_files) != len(stage_files):
        raise ValueError("The number of EEG files and stage files must match.")

    stage_mapping = {1: 'Wake', 2: 'NREM', 3: 'REM'}
    stage_colors = {stage: get_stage_color(stage) for stage in stage_mapping.values()}
    all_spectra = {stage: [] for stage in stage_mapping.values()}

    for eeg_file, stage_file in zip(eeg_files, stage_files):
        # Load EEG data from pickle file
        print(f"Loading EEG data from {eeg_file}...")
        with open(eeg_file, 'rb') as f:
            eeg_data = pickle.load(f)

        print(f"Loading sleep stage data from {stage_file}...")
        stages_data = pd.read_csv(stage_file)

        if channel not in eeg_data.columns:
            raise ValueError(f"Channel '{channel}' not found in EEG data.")

        # Extract the relevant channel and sleep stages
        eeg_signal = eeg_data[channel].values
        sleep_stages = stages_data['sleepStage'].values

        # Verify that EEG signal and stages align
        if len(eeg_signal) != len(sleep_stages):
            raise ValueError("EEG signal and sleep stage scoring lengths do not match.")

        renamed_stages = np.vectorize(stage_mapping.get)(sleep_stages)

        for stage in stage_mapping.values():
            # Extract data for the current sleep stage
            stage_indices = np.where(renamed_stages == stage)[0]
            stage_signal = eeg_signal[stage_indices]

            # Compute power spectrum using NeuroDSP's compute_spectrum
            freqs, power = compute_spectrum(stage_signal, fs=sampling_rate, method='welch', nperseg=sampling_rate * 2)

            # Limit to 0-40 Hz range
            freq_mask = (freqs >= 0.5) & (freqs <= 40)
            freqs = freqs[freq_mask]
            power = power[freq_mask]

            # Store power for averaging
            all_spectra[stage].append(power)

    # Rest of the function remains unchanged
    fig = plt.figure(figsize=(10, 6))

    for stage in stage_mapping.values():
        # Compute average and standard deviation of power spectra
        stage_powers = np.array(all_spectra[stage])
        avg_power = np.mean(stage_powers, axis=0)
        std_power = np.std(stage_powers, axis=0)

        # Plot average log power and shaded area for standard deviation
        plt.plot(freqs, 10 * np.log10(avg_power), label=stage, color=stage_colors[stage], linewidth=2.5)
        plt.fill_between(freqs, 10 * np.log10(avg_power - std_power), 10 * np.log10(avg_power + std_power), 
                         color=stage_colors[stage], alpha=0.3)

    plt.xlabel('Frequency (Hz)', fontsize=22)
    plt.ylabel('10*log(Power)', labelpad=20, fontsize=22)
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)

    # Set the x-axis to start at 0
    plt.xlim([0, 40])  # Limits the x-axis from 0 to 40 Hz
    plt.xticks(np.arange(0, 41, 5))  # Set x-ticks at 5 Hz intervals

    plt.legend(fontsize=20, frameon=False)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save the plot if output_file is provided
    if output_file:
        fig.savefig(output_file, dpi=600)
        if output_file.endswith('.png'):
            pdf_path = output_file[:-4] + '.pdf'
        else:
            pdf_path = f"{output_file}.pdf"
        fig.savefig(pdf_path, format='pdf')
        print(f"Plot saved to {output_file} and {pdf_path}")

    plt.show()

# Example usage (updated with pickle file paths):
plot_average_power_spectra(['/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-007_ses-01_recording-01.pkl', 
                            '/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-010_ses-01_recording-01.pkl',
                            '/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-011_ses-01_recording-01.pkl', 
                            '/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-015_ses-01_recording-01.pkl',
                            '/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-016_ses-02_recording-01.pkl', 
                            '/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-017_ses-01_recording-01.pkl'], 
                           ['/Volumes/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_512Hz.csv',
                            '/Volumes/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-010_ses-01_recording-01_time-0-69h_512Hz.csv',
                            '/Volumes/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-011_ses-01_recording-01_time-0-72h_512Hz.csv',
                            '/Volumes/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-49h_512Hz_stitched.csv',
                            '/Volumes/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_512Hz.csv',
                            '/Volumes/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_512Hz.csv'], 
                           channel='EEG1', sampling_rate=512, 
                           output_file="/Volumes/harris/volkan/sleep-profile/plots/frequency_power/frequency_logpower_all_sub_EEG1.png")