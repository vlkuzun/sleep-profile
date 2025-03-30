import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Define light and dark periods
light_period = (9, 21)
dark_period = (21, 9)

# Initialize lists to store results
id_list = []
period_list = []
avg_bout_length_list = []
bout_length_list = []
period_list_bout = []
id_list_bout = []

# Input CSV file names
num_files = int(input("Enter the number of CSV files: "))

for i in range(num_files):
    file_name = input(f"Enter CSV file name {i+1}: ")
    
    try:
        # Load CSV file
        df = pd.read_csv(file_name)
        
        # Convert Timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
        
        # Define period
        df['period'] = np.where((df['Timestamp'].dt.hour >= light_period[0]) & 
                                (df['Timestamp'].dt.hour < light_period[1]), 
                                'light', 'dark')
        
        # Filter for sleep stages (1, 2, 3)
        df = df[df['sleepStage'].isin([1, 2, 3])]
        
        # Calculate bout length
        df['bout_length'] = df.groupby(['period']).cumcount() + 1
        
        # Store bout lengths
        bout_length_list.extend(df['bout_length'].values)
        period_list_bout.extend(df['period'].values)
        id_list_bout.extend([file_name.split('.')[0]] * len(df))
        
        # Calculate average bout length for each period
        avg_bout_length = df.groupby('period')['bout_length'].mean()
        
        # Store results
        id_list.extend([file_name.split('.')[0]] * 2)
        period_list.extend(avg_bout_length.index)
        avg_bout_length_list.extend(avg_bout_length.values)
        
    except FileNotFoundError:
        print(f"File '{file_name}' not found. Skipping...")

# Create DataFrames with results
results_df = pd.DataFrame({'ID': id_list, 'period': period_list, 'avg_bout_length': avg_bout_length_list})
results_bout_df = pd.DataFrame({'ID': id_list_bout, 'period': period_list_bout, 'bout_length': bout_length_list})

# Plot individual points and average bars
plt.figure(figsize=(10, 6))
sns.stripplot(x='period', y='bout_length', data=results_bout_df, alpha=0.5, jitter=True)
sns.barplot(x='period', y='avg_bout_length', data=results_df, alpha=1, ci=None, color='black')
plt.xlabel('Period')
plt.ylabel('Bout Length')
plt.title('Bout Length in Light and Dark Periods')
plt.legend(title='Legend', labels=['Individual', 'Average'])
plt.savefig('bout_length_plot.png')

plt.show()