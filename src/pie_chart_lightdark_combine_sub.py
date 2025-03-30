import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

def create_pie_chart(stage_counts, output_path, title):
    # Define labels and sizes for the pie chart
    labels = ['Wake', 'NREM', 'REM']
    sizes = [stage_counts.get(1, 0), stage_counts.get(2, 0), stage_counts.get(3, 0)]

    # Create the pie chart with optimized font sizes
    plt.figure(figsize=(12, 9))  # Keep original figure size
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
            textprops={'fontsize': 32},  # Increased label font size
            colors=['#ff9999','#66b3ff','#99ff99'], 
            wedgeprops={'edgecolor': 'black', 'linewidth': 2.5})
    
    # Increase percentage value font size
    for text in plt.gca().texts:
        text.set_fontsize(32)  # Unified larger font size for all text elements
    
    plt.title(title, fontsize=40, pad=20)  # Larger title font size
    
    # Save with higher DPI for better quality
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
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

    all_light_phase_data = pd.DataFrame()
    all_dark_phase_data = pd.DataFrame()

    # Get directory containing CSV files
    input_dir = input("Enter the directory containing CSV files: ")
    if not os.path.exists(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.")
        return

    # Process all CSV files in the directory
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process...")

    # Process each CSV file
    for csv_file in csv_files:
        full_path = os.path.join(input_dir, csv_file)
        try:
            # Read the CSV file
            df = pd.read_csv(full_path)

            # Check if required columns exist
            if 'sleepStage' not in df.columns or 'Timestamp' not in df.columns:
                print(f"Error: Required columns not found in {csv_file}. Skipping this file.")
                continue

            print(f"Processing: {csv_file}")
            # Aggregate data for light and dark phases
            light_phase_data, dark_phase_data = aggregate_phases(df)

            # Combine with existing data
            all_light_phase_data = pd.concat([all_light_phase_data, light_phase_data], ignore_index=True)
            all_dark_phase_data = pd.concat([all_dark_phase_data, dark_phase_data], ignore_index=True)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    # Generate combined pie charts if data is available
    if not all_light_phase_data.empty and not all_dark_phase_data.empty:
        output_dir = input("Enter the output directory for the pie charts: ")
        if not os.path.exists(output_dir):
            print(f"Error: The directory '{output_dir}' does not exist. Please create it and try again.")
            return

        light_title = "Light"
        dark_title = "Dark"

        light_filename = os.path.join(output_dir, "combined_light_phase_pie_chart.png")
        dark_filename = os.path.join(output_dir, "combined_dark_phase_pie_chart.png")

        create_pie_chart(all_light_phase_data['sleepStage'].value_counts(), light_filename, light_title)
        create_pie_chart(all_dark_phase_data['sleepStage'].value_counts(), dark_filename, dark_title)
    else:
        print("No valid data was found to generate pie charts.")

if __name__ == "__main__":
    main()
