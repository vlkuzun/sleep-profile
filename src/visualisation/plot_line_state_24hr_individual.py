import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib import transforms
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
# Set global style for publication
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

    # Calculate the overall mean and SEM across all subjects for each ZT
    sleep_stages = ['wake_percent_mean', 'non_rem_percent_mean', 'rem_percent_mean']
    stage_titles = {'wake_percent_mean': 'Wake', 'non_rem_percent_mean': 'NREM', 'rem_percent_mean': 'REM'}
    stage_colors = {
        'wake_percent_mean': get_stage_color('Wake'),
        'non_rem_percent_mean': get_stage_color('NREM'),
        'rem_percent_mean': get_stage_color('REM'),
    }

    mean_df = df.groupby('ZT').mean(numeric_only=True)[sleep_stages]
    sem_df = df.groupby('ZT').sem(numeric_only=True)[sleep_stages]

    # Prepare subject palette for consistent coloring using Matplotlib
    subjects = sorted(df['subject'].unique())
    cmap = plt.get_cmap('tab10')
    subject_palette = {subj: cmap(idx % cmap.N) for idx, subj in enumerate(subjects)}
    line_styles = ['--', ':', '-.', (0, (3, 2))]

    fig, axes = plt.subplots(len(sleep_stages), 1, figsize=(14, 8), sharex=True)

    for ax, stage in zip(axes, sleep_stages):
        # Plot individual subjects with Matplotlib
        for idx, subject in enumerate(subjects):
            subject_data = df[df['subject'] == subject]
            ax.plot(
                subject_data['ZT'],
                subject_data[stage],
                color=subject_palette[subject],
                linewidth=0.8,
                linestyle=line_styles[idx % len(line_styles)],
                alpha=0.35
            )

        # Overlay mean and SEM for this stage
        ax.plot(mean_df.index, mean_df[stage], color=stage_colors[stage], linewidth=3, label='Mean')
        ax.fill_between(
            mean_df.index,
            mean_df[stage] - sem_df[stage],
            mean_df[stage] + sem_df[stage],
            color=stage_colors[stage],
            alpha=0.25,
            label='SEM'
        )

        ax.set_ylim(-2, 102)
        ax.set_yticks(range(0, 105, 20))
        ax.set_ylabel('Percent (%)')
        ax.set_title(stage_titles[stage])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add phase indicator bar only to the bottom subplot
        if ax is axes[-1]:
            bar_height = 0.08  # fraction of axes height
            blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.add_patch(Rectangle((0, -bar_height), 1, bar_height, transform=blended,
                           color='#FFD1A1', alpha=0.8, lw=0, clip_on=False))
            ax.add_patch(Rectangle((1, -bar_height), 11, bar_height, transform=blended,
                           color='orange', alpha=0.8, lw=0, clip_on=False))
            ax.add_patch(Rectangle((12, -bar_height), 1, bar_height, transform=blended,
                           color='#C0C0C0', alpha=0.8, lw=0, clip_on=False))
            ax.add_patch(Rectangle((13, -bar_height), 10, bar_height, transform=blended,
                           color='gray', alpha=0.8, lw=0, clip_on=False))

        # Ensure x-axis limits align with integer ZT range
        xticks_range = range(int(mean_df.index.min()), int(mean_df.index.max()) + 1)
        ax.set_xlim(min(xticks_range), max(xticks_range))
        ax.set_xticks(xticks_range)
        ax.set_xticklabels([int(x) for x in xticks_range])

    axes[-1].set_xlabel('Zeitgeber time (ZT)')
    axes[-1].set_xticks(sorted(df['ZT'].unique()))
    axes[-1].tick_params(axis='x', pad=10)

    # Remove legend to keep focus on mean lines

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save before showing to avoid blank exports
    if output_file:
        fig.savefig(output_file, dpi=600, bbox_inches='tight')
        if output_file.endswith('.png'):
            pdf_path = output_file.replace('.png', '.pdf')
        else:
            pdf_path = f"{output_file}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to: {output_file} and {pdf_path}")

    plt.show()
    plt.close(fig)

# Get input file path and output file path
input_file = input("Enter the path of the combined CSV file: ")
output_file = input("Enter the output file path for the plot (e.g., path/to/plot.png): ")
plot_combined_sleep_data(input_file, output_file)
