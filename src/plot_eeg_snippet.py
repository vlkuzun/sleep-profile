import pandas as pd
import matplotlib.pyplot as plt

def plot_eeg_signal(file_path, sampling_frequency, time_range, channel="EEG1", downsample_factor=1):
    """
    Plots EEG signal for a specified time range with an optional downsampling factor.

    Parameters:
        file_path (str): Path to the CSV file containing EEG signal data.
        sampling_frequency (float): Frequency of EEG signal acquisition (Hz).
        time_range (tuple): Start and end time (in seconds) for the signal to be plotted.
        channel (str): The column name of the EEG channel to plot (default is 'EEG1').
        downsample_factor (int): Factor by which to downsample the signal (default is 1, no downsampling).

    The CSV file is expected to have columns for EEG signal values.
    """
    try:
        # Load the EEG data from CSV
        eeg_data = pd.read_csv(file_path)

        if channel not in eeg_data.columns:
            raise ValueError(f"The specified channel '{channel}' is not in the CSV file.")

        signal = eeg_data[channel].values

        # Compute time axis based on sampling frequency
        total_samples = len(signal)
        total_time = total_samples / sampling_frequency
        time = [i / sampling_frequency for i in range(total_samples)]

        # Extract indices for the time range
        start_index = int(time_range[0] * sampling_frequency)
        end_index = int(time_range[1] * sampling_frequency)

        if start_index < 0 or end_index > total_samples or start_index >= end_index:
            raise ValueError("Invalid time range specified.")

        # Slice signal and time for the specified range
        time_segment = time[start_index:end_index]
        signal_segment = signal[start_index:end_index]

        # Downsample the signal if specified
        if downsample_factor > 1:
            time_segment = time_segment[::downsample_factor]
            signal_segment = signal_segment[::downsample_factor]

        # Adjust time segment to start at 0
        adjusted_time_segment = [t - time_segment[0] for t in time_segment]

        # Plot the signal
        plt.figure(figsize=(16, 3))
        plt.plot(adjusted_time_segment, signal_segment, color='black')
        plt.title('REM', fontsize=26)
        plt.xlabel("Time (s)", fontsize=22)

        # Set x-ticks to start at 0, repeat every second, and include the last time in the range
        max_time = round(adjusted_time_segment[-1])
        x_ticks = list(range(0, max_time + 1))
        plt.xticks(x_ticks, [str(x) for x in x_ticks], fontsize=22)

        # Remove all spines
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        # Remove y-axis tick labels and values
        plt.yticks([])

        plt.show()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
plot_eeg_signal('/Volumes/harris/somnotate/to_score_set/to_score_csv_files/sub-015_ses-01_recording-01_time-0-20h.csv', sampling_frequency=512, time_range=(20842, 20847), channel='EEG2', downsample_factor=5)