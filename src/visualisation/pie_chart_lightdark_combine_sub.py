import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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

from src.stage_colors import get_stage_palette

# Match publication styling used by 24 h line plots
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

STAGE_CODES = [1, 2, 3]
STAGE_LABELS = ['Wake', 'NREM', 'REM']
STAGE_PALETTE = get_stage_palette(STAGE_LABELS)

def create_pie_chart(stage_counts, output_path, title):
    sizes = [stage_counts.get(code, 0) for code in STAGE_CODES]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=STAGE_LABELS,
        autopct='%1.1f%%',
        startangle=140,
        colors=STAGE_PALETTE,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 12, 'color': '#111111'}
    )

    for txt in texts + autotexts:
        txt.set_fontsize(12)

    ax.set_title(title, pad=16)
    ax.axis('equal')

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        if output_path.endswith('.png'):
            pdf_path = output_path.replace('.png', '.pdf')
        else:
            pdf_path = f"{output_path}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Pie chart saved to: {output_path} and {pdf_path}")

    plt.close(fig)

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
