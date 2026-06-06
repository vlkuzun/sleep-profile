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
# Match global publication style with individual plot
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

def plot_combined_sleep_data(input_file, output_file):
    # Load the combined CSV file with all subjects' data
    df = pd.read_csv(input_file)

    # Ensure data is sorted by ZT for proper plotting
    df = df.sort_values(by='ZT')

    sleep_stages = ['wake_percent_mean', 'non_rem_percent_mean', 'rem_percent_mean']
    stage_titles = {'wake_percent_mean': 'Wake', 'non_rem_percent_mean': 'NREM', 'rem_percent_mean': 'REM'}
    stage_colors = {
        'wake_percent_mean': get_stage_color('Wake'),
        'non_rem_percent_mean': get_stage_color('NREM'),
        'rem_percent_mean': get_stage_color('REM'),
    }

    mean_df = df.groupby('ZT').mean(numeric_only=True)[sleep_stages]
    sem_df = df.groupby('ZT').sem(numeric_only=True)[sleep_stages]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Add full-height background phase shading behind the traces.
    span_kwargs = {
        'linewidth': 0,
        'edgecolor': 'none',
        'antialiased': False,
        'zorder': 0,
    }
    ax.axvspan(0, 1, color='#FFD1A1', alpha=0.8, **span_kwargs)
    ax.axvspan(1, 12, color='orange', alpha=0.5, **span_kwargs)
    ax.axvspan(12, 13, color='#C0C0C0', alpha=0.5, **span_kwargs)
    ax.axvspan(13, 23, color='gray', alpha=0.5, **span_kwargs)

    # Plot mean and SEM for each sleep stage with consistent styling
    for stage in sleep_stages:
        ax.plot(
            mean_df.index,
            mean_df[stage],
            color=stage_colors[stage],
            linewidth=3,
            label=stage_titles[stage],
            zorder=3
        )
        ax.fill_between(
            mean_df.index,
            mean_df[stage] - sem_df[stage],
            mean_df[stage] + sem_df[stage],
            color=stage_colors[stage],
            alpha=0.25,
            zorder=2
        )

    xticks_range = range(int(mean_df.index.min()), int(mean_df.index.max()) + 1)
    ax.set_xlim(min(xticks_range), max(xticks_range))
    ax.set_xticks(xticks_range)
    ax.set_xticklabels([int(x) for x in xticks_range])
    ax.tick_params(axis='x', pad=12)

    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))

    ax.set_xlabel('Zeitgeber time (ZT)')
    ax.set_ylabel('Percent (%)')

    # Legend matches individual plot style (subjects absent so keep concise)
    ax.legend(frameon=False, loc='upper left')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_file = output_file.strip()

    if output_file:
        fig.savefig(output_file, dpi=600, bbox_inches='tight')
        if output_file.endswith('.png'):
            pdf_path = output_file.replace('.png', '.pdf')
        else:
            pdf_path = f"{output_file}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to: {output_file} and {pdf_path}")
    else:
        print("No output path provided; displaying plot instead.")
        plt.show()


# Get input file path and output file path
input_file = input("Enter the path of the combined CSV file: ")
output_file = input("Enter the output file path for the plot (optional; press Enter to display): ")
plot_combined_sleep_data(input_file, output_file)
