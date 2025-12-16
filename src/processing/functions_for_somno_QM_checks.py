import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def match_length_csv_files(df1, df2):
    '''
    Check if the two CSV files have the same number of samples. If not, truncate the longer file to match the length of the shorter file.
    Input:
        df1: Path to the first CSV file
        df2: Path to the second CSV file
    Output:
        df1: Truncated CSV file 1
        df2: Truncated CSV file 2
    '''
    
    len_csv1 = len(df1)
    len_csv2 = len(df2)

    if len_csv1 != len_csv2:
        print(f"Length mismatch: CSV1 has {len_csv1} samples, CSV2 has {len_csv2} samples.")
        if len_csv1 > len_csv2:
            excess_rows = len_csv1 - len_csv2
            df1 = df1[:-excess_rows]
            print(f"CSV1 truncated by {excess_rows} samples to match length of CSV2")
        else:
            excess_rows = len_csv2 - len_csv1
            df2 = df2[:-excess_rows]
            print(f"CSV2 truncated by {excess_rows} samples to match length of CSV1 ") 
    
    assert len(df1) == len(df2), "Length of CSV1 does not match length of CSV2 after truncation"

    return df1, df2


def compare_csv_files(df1, df2):
    ''' 
    Compare the sleep stages from two CSV files.
    Input:
        df1: Path to the first CSV file
        df2: Path to the second CSV file
    Output:
        percentage_similarity: Percentage of samples where the sleep stages match between the two CSV files
    '''

    matches = df1['sleepStage'] == df2['sleepStage'] # element-wise comparison of the two columns
    percentage_similarity = np.mean(matches) * 100
    print(f"Percentage similarity: {percentage_similarity:.2f}%")

    return percentage_similarity

def rename_file(file_path):
    '''
    Label the files based on their filenames.
    Input:
        file_path: Path to the file
    Output:
        label: Label for the file
    '''
    filename = os.path.basename(file_path)
    if "automated" in filename:
        return "somnotate"
    if "fp" in filename:
        return "fp"
    if "vu" in filename:
        return "vu"
    if "BH" in filename:
        return "bh"
    else:
        return "control"

def compare_csv_files_by_stage(df_manual, df_somnotate, stage_value):
    ''' 
    Compare specific sleep stages between a manual CSV file and the somnotate CSV file.
    Input:
        df_manual: DataFrame for the manual file
        df_somnotate: DataFrame for the somnotate file
        stage_value: The value of the sleep stage to filter by (1: awake, 2: non-REM, 3: REM, etc.)
    Output:
        percentage_similarity: Percentage similarity between the manual and somnotate annotations for this sleep stage
    '''

    # Get indices in the manual annotations where the sleep stage is equal to a given stage_value (e.g., 1 for 'awake')
    manual_stage_indices = df_manual[df_manual['sleepStage'] == stage_value].index

    # Get the corresponding sleep stages in somnotate based on these indices
    somnotate_stage_at_indices = df_somnotate.loc[manual_stage_indices, 'sleepStage']

    # Get the manual sleep stages at those same indices (should all be stage_value)
    manual_stage_at_indices = df_manual.loc[manual_stage_indices, 'sleepStage']
    assert manual_stage_at_indices.all() == stage_value, f"Manual sleep stages at indices {manual_stage_indices} are not all {stage_value}"

    # Compare the two (manual vs somnotate) for those indices
    matches = manual_stage_at_indices == somnotate_stage_at_indices
    percentage_similarity = np.mean(matches) * 100

    print(f"Percentage similarity for sleep stage {stage_value} (manual vs somnotate): {percentage_similarity:.2f}%")
    
    return percentage_similarity

def compute_confusion_matrix_by_stage(df_manual, df_somnotate, stages):
    ''' 
    Compute confusion matrix for misclassification of sleep stages, after checking for length mismatch.
    Input:
        df_manual: DataFrame for the manual file
        df_somnotate: DataFrame for the somnotate file
        stages: Dictionary mapping sleep stage names to their respective values (e.g., {"awake": 1, "non-REM": 2, "REM": 3})
    Output:
        confusion_matrix: A matrix with counts of misclassifications between stages
    '''
    # Ensure both DataFrames are of equal length
    df_manual, df_somnotate = match_length_csv_files(df_manual, df_somnotate) 

    # Initialize the confusion matrix (N x N, where N is the number of stages)
    num_stages = len(stages)
    confusion_matrix = np.zeros((num_stages, num_stages))  # Rows: manual, Columns: somnotate

    for manual_stage_name, manual_stage_value in stages.items():

        manual_stage_indices = df_manual[df_manual['sleepStage'] == manual_stage_value].index # get the indices where the manual stage is equal to the current stage value (e.g., 'awake')
        somnotate_stage_at_indices = df_somnotate.loc[manual_stage_indices, 'sleepStage'] # get the somnotate stage at the same indices

        for somnotate_stage_value in somnotate_stage_at_indices:
            confusion_matrix[stages[manual_stage_name] - 1, somnotate_stage_value - 1] += 1

    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, labels, title="Sleep Stage Confusion Matrix"):
    '''
    Plot the confusion matrix showing misclassifications between sleep stages.
    Input:
        confusion_matrix: A NxN matrix with misclassification counts
        labels: List of sleep stages (e.g., ['awake', 'non-REM', 'REM'])
        title: Title for the plot
    '''
    # Normalize the confusion matrix by dividing by row sums to get percentages
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = confusion_matrix / row_sums

    # Create a DataFrame for the heatmap
    df = pd.DataFrame(normalized_matrix, index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", cbar=True, linewidths=0.5)
    plt.title(title)
    plt.xlabel('Somnotate Annotation')
    plt.ylabel('Manual Annotation')
    plt.tight_layout()
    plt.show()


def get_bout_durations(df, sampling_rate, df_name):
    '''
    Calculate the duration of bouts for each sleep stage in a CSV file.
    Input:
        df: DataFrame for the CSV file
        sampling_rate: Sampling rate of the data in Hz
    Output:
        bout_durations_with_stage_all: Dictionary with DataFrame for each CSV file containing bout durations and corresponding sleep stages
    '''
        
    print(f"Type of input df: {type(df)}")  # Debug: Check type of df

    # if 'sleepStage' not in df.columns:
    #     raise ValueError("The 'sleepStage' column is missing from the DataFrame.")
    
    bout_durations_with_stage = []
    bout_durations = []
    bout_stages = []
    bout_durations_with_stage_all = {}

    stage_changes = np.where(df['sleepStage'].diff() != 0)[0]

    previous_time = 0 
    previous_stage = df['sleepStage'].iloc[0]
    

    for stage_change in stage_changes:
        bout_duration = (stage_change - previous_time) / sampling_rate
        bout_durations.append(bout_duration)
        bout_stages.append(previous_stage)

        previous_time = stage_change
        previous_stage = df['sleepStage'].iloc[stage_change]

    final_bout_duration = (len(df) - previous_time) / sampling_rate
    bout_durations.append(final_bout_duration)  # Add the duration of the last bout
    bout_stages.append(previous_stage) # Add the stage of the last bout

    bout_durations_with_stage = pd.DataFrame({'BoutDuration': bout_durations, 'SleepStage': bout_stages})
    bout_durations_with_stage_all[df_name] = bout_durations_with_stage

    return bout_durations_with_stage_all

def get_stage_durations(bout_durations_with_stage_all):
    '''
    Extract bout durations for awake, NREM, and REM stages from all dataframes.
    Input:
        bout_durations_with_stage_all: Dictionary of DataFrames containing bout durations and sleep stages
    Output:
        bout_durations_awake, bout_durations_nrem, bout_durations_rem: Dictionaries of bout durations for each stage
    '''
    
    bout_durations_awake = {}
    bout_durations_nrem = {}
    bout_durations_rem = {}

    for df_name, df in bout_durations_with_stage_all.items():
        # Extract bout durations for specific sleep stages (1: awake, 2: NREM, 3: REM)
        awake_durations = df.loc[df['SleepStage'] == 1, 'BoutDuration'].tolist()
        nrem_durations = df.loc[df['SleepStage'] == 2, 'BoutDuration'].tolist()
        rem_durations = df.loc[df['SleepStage'] == 3, 'BoutDuration'].tolist()

        bout_durations_awake[df_name] = awake_durations
        bout_durations_nrem[df_name] = nrem_durations
        bout_durations_rem[df_name] = rem_durations

    return bout_durations_awake, bout_durations_nrem, bout_durations_rem

def perform_anova(bout_durations_dict, sleep_stage_label):

    if sleep_stage_label == 'All stages':
        print("No sleep stage label provided. Performing ANOVA on all sleep stages combined.")
        # Extract all bout durations from the dictionaries for each dataframe
        all_durations = {df_name: df['BoutDuration'].tolist() for df_name, df in bout_durations_dict.items()}
        # Convert the dictionary into a list of lists (each list corresponds to bout durations from one dataframe)
        data = [durations for durations in all_durations.values()]

    else:
        print(f"Performing ANOVA for {sleep_stage_label}")
        data = [durations for durations in bout_durations_dict.values()]
    
    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*data)
    
    print(f"ANOVA results for {sleep_stage_label}:")
    print(f"F-statistic: {f_stat}")
    print(f"P-value: {p_value}")
    
    # Interpretation
    if p_value < 0.05:
        print(f"Significant differences found between dataframes for {sleep_stage_label} (p < 0.05)\n")
    
    return f_stat, p_value

def tukey_test(bout_durations_dict, sleep_stage_label):
    '''
    Perform Tukey's post-hoc test to compare the means of bout durations between different dataframes.
    Input:
        bout_durations_dict: Dictionary of bout durations for each dataframe

    Output:
        tukey_results: Results of the Tukey's HSD test
    '''

    # Combine all bout durations into a single list and add labels for dataframes
    durations = []
    labels = []
    
    for df_name, duration_list in bout_durations_dict.items():
        if sleep_stage_label == 'All stages':
            if isinstance(duration_list, pd.DataFrame):
                duration_list = duration_list['BoutDuration'].tolist()
            else:
                raise ValueError(f"Expected DataFrame for {df_name} when sleep_stage_label is None, got {type(duration_list)}")
        else:
            duration_list = duration_list    
        
        durations.extend(duration_list)
        labels.extend([df_name] * len(duration_list))
    
    
    # Perform Tukey's HSD
    data = pd.DataFrame({'BoutDuration': durations, 'DataFrame': labels})
    tukey = pairwise_tukeyhsd(endog=data['BoutDuration'], groups=data['DataFrame'], alpha=0.05)
    print(tukey)
    
    return tukey       


def plot_bout_duration_histograms_with_significance(bout_durations_dict, sleep_stage_label):
    plt.figure(figsize=(12, 6))
    labels = list(bout_durations_dict.keys())
    
    # Calculate means and standard errors
    means = []
    ses = []
    for df_name in labels:
        data = bout_durations_dict[df_name]
        if sleep_stage_label == 'All stages':
                data = data['BoutDuration'].tolist()
        
        means.append(np.mean(data))
        ses.append(stats.sem(data) if len(data) > 1 else 0)  # SEM, avoid division by zero

    x = np.arange(len(labels))  # X-axis positions for bars
    y_max = max(means) + max(ses)  # Max y value for positioning brackets
    
    # Plot bars with error bars (means and standard errors)
    plt.bar(x, means, yerr=ses, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    plt.xticks(x, labels)
    plt.ylabel('Mean Bout Duration (seconds)')
    
    if sleep_stage_label:
        plt.title(f'Bout Durations for {sleep_stage_label}')
    else:
        plt.title('Overall Bout Durations Across DataFrames')
    
    plt.ylim(0, y_max + 0.2 * y_max)  # Adjust y-limits

    # Perform Tukey's HSD if multiple dataframes are available
    if len(labels) > 1:
        tukey = tukey_test(bout_durations_dict, sleep_stage_label)
        comparisons = tukey._results_table.data[1:]  # Extract results from Tukey's test
        significance_threshold = 0.05  # Significance level for stars

        # Annotate significance stars on pairwise comparisons
        for comparison in comparisons:
            group1, group2, meandiff, p_adj, lower, upper, reject = comparison
            if p_adj < significance_threshold:  # Only annotate significant differences
                idx1 = labels.index(group1)
                idx2 = labels.index(group2)
                label_position = (idx1 + idx2) / 2  # Position between the two bars

                # Calculate y_position for the bracket and significance label
                y_position = y_max + 0.1 * y_max
                plt.text(label_position, y_position, '*', ha='center', va='bottom', fontsize=12)

                # Add brackets
                plt.plot([idx1, idx1, idx2, idx2], [y_max, y_position, y_position, y_max], lw=1.5, color='black')

    f_stat, p_value = perform_anova(bout_durations_dict, sleep_stage_label)
    
    # Adjust the position of ANOVA results to avoid overlap with significance stars
    plt.text(0.5, 0.75, f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}", ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.show()


def count_transitions(df, df_name):
    ''' 
    Calculate the number of transitions between sleep stages in a CSV file.
    Input:
        df: DataFrame for the CSV file
    Output
        n_transitions_all: Dictionary with the number of transitions for each CSV file
    '''

    n_transitions_all = {}
    stage_changes = np.where(df['sleepStage'].diff() != 0)[0]
    n_transitions = len(stage_changes)
    n_transitions_all[df_name] = n_transitions
    print(f'The number of transitions for {df_name} is {n_transitions}')

    return n_transitions_all

def count_REM_to_non_REM_transitions(df, df_name):
    ''' 
    Calculate the number of transitions from non-REM to REM sleep stages in a CSV file.
    Input:
        df: DataFrame for the CSV file
    Output
        n_incorrect_transitions_all : Dictionary with the number of non-REM to REM transitions for each CSV file                  
    '''

    n_incorrect_transitions_all = {}
    stage_changes = np.where(df['sleepStage'].diff() != 0)[0]
    n_incorrect_transitions = 0
    for i in range(len(stage_changes)-1):
        if df['sleepStage'].iloc[stage_changes[i]] == 3 and df['sleepStage'].iloc[stage_changes[i+1]] == 2:
            n_incorrect_transitions += 1

    n_incorrect_transitions_all[df_name] = n_incorrect_transitions
    print(f'The number of non-REM to REM transitions for {df_name} is {n_incorrect_transitions}')



    return n_incorrect_transitions_all

def count_REM_to_awake_transitions(df, df_name):
    ''' 
    Calculate the number of transitions from REM to awake sleep stages in a CSV file.
    Input:
        df: DataFrame for the CSV file
    Output
        n_REM_to_awake_transitions_all : Dictionary with the number of REM to awake transitions for each CSV file                  
    '''

    n_REM_to_awake_transitions_all = {}
    stage_changes = np.where(df['sleepStage'].diff() != 0)[0]
    n_REM_to_awake_transitions = 0
    for i in range(len(stage_changes)-1):
        if df['sleepStage'].iloc[stage_changes[i]] == 3 and df['sleepStage'].iloc[stage_changes[i+1]] == 1:
            n_REM_to_awake_transitions += 1

    n_REM_to_awake_transitions_all[df_name] = n_REM_to_awake_transitions
    print(f'The number of REM to awake transitions for {df_name} is {n_REM_to_awake_transitions}')

    return n_REM_to_awake_transitions_all

def count_non_REM_to_awake_transitions(df, df_name):
    '''
    Calculate the number of transitions from non-REM to awake sleep stages in a CSV file.
    Input:
        df: DataFrame for the CSV file
    Output:
        n_non_REM_to_awake_transitions_all : Dictionary with the number of non-REM to awake transitions for each CSV file
    '''

    n_non_REM_to_awake_transitions_all = {}
    stage_changes = np.where(df['sleepStage'].diff() != 0)[0]
    n_non_REM_to_awake_transitions = 0
    for i in range(len(stage_changes)-1):
        if df['sleepStage'].iloc[stage_changes[i]] == 2 and df['sleepStage'].iloc[stage_changes[i+1]] == 1:
            n_non_REM_to_awake_transitions += 1

    n_non_REM_to_awake_transitions_all[df_name] = n_non_REM_to_awake_transitions    
    print(f'The number of non-REM to awake transitions for {df_name} is {n_non_REM_to_awake_transitions}')

    return n_non_REM_to_awake_transitions_all


def plot_transitions(n_transitions_all):
    ''' 
    Plot the number of transitions between sleep stages for each CSV file.
    Input:
        n_transitions_all: Dictionary with the number of transitions for each CSV file
    '''

    df_names = list(n_transitions_all.keys())
    print(df_names)
    n_transitions_values = [list(item.values())[0] for item in n_transitions_all.values()]
    print(n_transitions_values)

    plt.bar(df_names, n_transitions_values, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    plt.ylim(0, 180)
    plt.ylabel('Number of transitions')
    plt.show()


        















        


    


    
    









    

















