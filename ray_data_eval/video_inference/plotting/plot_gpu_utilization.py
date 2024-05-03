import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGRATIO = 3 / 5
FIGWIDTH = 4  # inches
FIGHEIGHT = FIGWIDTH * FIGRATIO
FIGSIZE = (FIGWIDTH, FIGHEIGHT)

plt.rcParams.update(
    {
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.figsize": FIGSIZE,
        "figure.dpi": 300,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": True,
    }
)

COLORS = sns.color_palette("Paired")
sns.set_style("ticks")
sns.set_palette(COLORS)

MICROSECS_IN_SEC = 1_000_000


def get_gpu_spans(timeline_file: str):
    df = pd.read_json(timeline_file)
    start_time = df[df["cat"].str.startswith("task::MapBatches(produce_video_slices")]["ts"].min()
    print(start_time)
    gpu_df = df[df["cat"] == "task::MapBatches(Classifier)"]
    spans = []
    for _, row in gpu_df.iterrows():
        span_start = (row["ts"] - start_time) / MICROSECS_IN_SEC
        span_end = span_start + row["dur"] / MICROSECS_IN_SEC
        spans.append((span_start, span_end))
    spans.sort()
    print(spans)
    return spans


def main():
    data = {
        "Without Dynamic Repartitioning": get_gpu_spans("static_partitioning.json"),
        "With Dynamic Repartitioning": get_gpu_spans("dynamic_partitioning.json"),
    }
    fig, axes = plt.subplots(len(data), 1, sharex=True)
    if len(data) == 1:
        axes = [axes]

    # Plot each timeline on its subplot
    for i, (ax, (title, spans)) in enumerate(zip(axes, data.items())):
        color = COLORS[i * 2]
        for start, end in spans:
            ax.axvspan(start, end, color=color)

        ax.set_title(title)
        ax.set_ylabel("GPU Util.")
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 100])
        ax.set_yticklabels([])
        ax.grid(axis="y", linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    max_time = 180

    # Set shared x-axis properties
    axes[-1].set_xlabel("Time (s)")  # Only set x-label on the last subplot
    axes[-1].set_xlim(0, max_time)
    axes[-1].set_xticks(range(0, max_time + 1, 30))

    # Adjust layout, save and show the plot
    plt.tight_layout()
    plt.subplots_adjust(hspace=1)

    output_filename = "repartitioning_timelines.pdf"
    print(output_filename)
    plt.savefig(output_filename, bbox_inches="tight")


if __name__ == "__main__":
    main()
