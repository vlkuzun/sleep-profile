import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import transforms

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
        'wake_percent_mean': '#E69F00',
        'non_rem_percent_mean': '#56B4E9',
        'rem_percent_mean': '#CC79A7'
    }

    mean_df = df.groupby('ZT').mean(numeric_only=True)[sleep_stages]
    sem_df = df.groupby('ZT').sem(numeric_only=True)[sleep_stages]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot mean and SEM for each sleep stage with consistent styling
    for stage in sleep_stages:
        ax.plot(
            mean_df.index,
            mean_df[stage],
            color=stage_colors[stage],
            linewidth=3,
            label=stage_titles[stage]
        )
        ax.fill_between(
            mean_df.index,
            mean_df[stage] - sem_df[stage],
            mean_df[stage] + sem_df[stage],
            color=stage_colors[stage],
            alpha=0.25
        )

    # Add phase indicator bar below axis similar to individual plot
    bar_height = 0.04
    blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.add_patch(Rectangle((0, -bar_height), 1, bar_height, transform=blended,
                           color='#FFD1A1', alpha=0.8, lw=0, clip_on=False))
    ax.add_patch(Rectangle((1, -bar_height), 11, bar_height, transform=blended,
                           color='orange', alpha=0.8, lw=0, clip_on=False))
    ax.add_patch(Rectangle((12, -bar_height), 1, bar_height, transform=blended,
                           color='#C0C0C0', alpha=0.8, lw=0, clip_on=False))
    ax.add_patch(Rectangle((13, -bar_height), 10, bar_height, transform=blended,
                           color='gray', alpha=0.8, lw=0, clip_on=False))

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

    if output_file:
        fig.savefig(output_file, dpi=600, bbox_inches='tight')
        if output_file.endswith('.png'):
            pdf_path = output_file.replace('.png', '.pdf')
        else:
            pdf_path = f"{output_file}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to: {output_file} and {pdf_path}")


# Get input file path and output file path
input_file = input("Enter the path of the combined CSV file: ")
output_file = input("Enter the output file path for the plot (e.g., path/to/plot.png): ")
plot_combined_sleep_data(input_file, output_file)
