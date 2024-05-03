import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGRATIO = 2 / 5
FIGWIDTH = 10  # inches
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


def load_timeline(filename):
    df = pd.read_json(filename)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    events = []
    start_time = 0
    EPS = 1e-6

    def add_events(row, event_name):
        nonlocal start_time, events
        events.append((row.name - start_time, event_name, 0))
        events.append((row.name - start_time + EPS, event_name, 1))
        events.append((row.name - start_time + row["dur"], event_name, 0))
        events.append((row.name - start_time + row["dur"] + EPS, event_name, -1))

    for _, row in df.iterrows():
        # if row["cat"].startswith("task::MapBatches(produce_video_slices"):
        #     start_time = row.name
        #     add_events(row, "Produce")
        # elif row["cat"] == "task::MapBatches(preprocess_video)":
        #     add_events(row, "Preprocess")
        # elif row["cat"] == "task::MapBatches(Classifier)":
        #     add_events(row, "Classifier")
        if row["cat"] == "task::ReadRange->MapBatches(produce)":
            if start_time == 0:
                start_time = row.name
            add_events(row, "Producer")
        elif row["cat"] == "task::MapBatches(consume)":
            add_events(row, "Consumer")

    edf = pd.DataFrame(events, columns=["Time", "Event", "Change"])
    edf["Time"] /= 1_000_000
    edf.sort_values("Time", inplace=True)
    dfs = {event: group for event, group in edf.groupby("Event")}

    final_df = pd.DataFrame({"Time": []})
    for event, df in dfs.items():
        df["Cumulative"] = df["Change"].cumsum()
        print(event, df)
        final_df = final_df.merge(
            df[["Time", "Cumulative"]].rename(columns={"Cumulative": event}),
            on="Time",
            how="outer",
        )
    final_df.sort_values("Time", inplace=True)
    final_df.fillna(method="ffill", inplace=True)
    print(final_df)

    # make a stack plot
    fig, ax = plt.subplots()
    ax.stackplot(
        final_df["Time"],
        final_df.drop(columns="Time").T,
        labels=final_df.columns[1:],
        baseline="zero",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Parallelism")
    ax.legend()
    plt.savefig(f"{filename}.png")


# load_timeline("dynamic_partitioning.json")
load_timeline("mb_timeline.json")
