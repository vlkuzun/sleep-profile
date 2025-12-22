import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Set Seaborn theme
sns.set_theme(style="whitegrid")

def create_pie_chart(stage_counts, output_path, title):
    # Define labels and sizes for the pie chart
    labels = ['Wake', 'Non-REM', 'REM']
    sizes = [stage_counts.get(1, 0), stage_counts.get(2, 0), stage_counts.get(3, 0)]

    # Create the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
            colors=['#ff9999','#66b3ff','#99ff99'], wedgeprops={'edgecolor': 'black'})
    plt.title(title)
    
    # Save and close the chart
    plt.savefig(output_path)
    plt.close()
    print(f"Pie chart saved to: {output_path}")

def aggregate_phases(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Filter for light phase (09:00 to 21:00)
    light_phase_data = df[(df['Timestamp'].dt.time >= pd.Timestamp('09:00').time()) & 
                          (df['Timestamp'].dt.time < pd.Timestamp('21:00').time())]
    
    # Filter for dark phase (21:00 to 09:00)
    dark_phase_data = df[(df['Timestamp'].dt.time >= pd.Timestamp('21:00').time()) | 
                         (df['Timestamp'].dt.time < pd.Timestamp('09:00').time())]
    
    return light_phase_data, dark_phase_data

def main():
    print("Welcome to the Sleep Stage Pie Chart Generator!")

    # Loop to process multiple CSV files
    while True:
        csv_file = input("Enter the path of a CSV file (or type 'done' to finish): ")
        if csv_file.lower() == 'done':
            break
        elif os.path.exists(csv_file):
            # Get the output directory path
            output_dir = input("Enter the output directory for the pie charts: ")
            if not os.path.exists(output_dir):
                print(f"Error: The directory '{output_dir}' does not exist. Please try again.")
                continue

            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)

                # Check if 'sleepStage' column exists
                if 'sleepStage' not in df.columns:
                    print(f"Error: 'sleepStage' column not found in {csv_file}. Skipping this file.")
                    continue

                # Get metadata from the user
                subject = input("Enter subject: ")
                session = input("Enter session: ")
                recording = input("Enter recording: ")
                extra_info = input("Enter extra information: ")

                # Ask the user for custom or default title for charts
                use_default_title = input("Use subject and session as titles? (yes/no): ").strip().lower()
                if use_default_title == "yes":
                    light_title = f'Sleep Stage Distribution for {subject}_{session} (Light Phase)'
                    dark_title = f'Sleep Stage Distribution for {subject}_{session} (Dark Phase)'
                else:
                    light_title = input("Enter a custom title for the light phase chart: ")
                    dark_title = input("Enter a custom title for the dark phase chart: ")

                # Aggregate data for light and dark phases
                light_phase_data, dark_phase_data = aggregate_phases(df)

                # Generate file names for pie charts
                light_filename = os.path.join(output_dir, f"{subject}_{session}_{recording}_{extra_info}_light_phase_pie_chart.png")
                dark_filename = os.path.join(output_dir, f"{subject}_{session}_{recording}_{extra_info}_dark_phase_pie_chart.png")

                # Create and save pie charts for each phase
                create_pie_chart(light_phase_data['sleepStage'].value_counts(), light_filename, light_title)
                create_pie_chart(dark_phase_data['sleepStage'].value_counts(), dark_filename, dark_title)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        else:
            print(f"Error: The file at '{csv_file}' does not exist. Please try again.")

if __name__ == "__main__":
    main()
