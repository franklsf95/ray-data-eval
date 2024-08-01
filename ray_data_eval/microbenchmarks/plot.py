import matplotlib.pyplot as plt
import seaborn as sns

FIGRATIO = 3 / 4
FIGWIDTH = 3.335  # inches
FIGHEIGHT = FIGWIDTH * FIGRATIO
FIGSIZE = (FIGWIDTH, FIGHEIGHT)

BIG_SIZE = 12
plt.rcParams.update(
    {
        "figure.figsize": FIGSIZE,
        "figure.dpi": 300,
        "text.usetex": True,
    }
)

COLORS = sns.color_palette("pastel")
sns.set_style("ticks")
sns.set_palette(COLORS)

plt.rc("font", size=BIG_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIG_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIG_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=BIG_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=BIG_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=BIG_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIG_SIZE)  # fontsize of the figure title

# Data
frameworks = ["Radar", "Spark", "Spark\nStreaming", "Flink", "tf.data"]
unlimited_memory = [209, 326, 291.76, 475.53, 305]
limited_memory = [211, 843.88, 890.64, 624.6, 549.6]  # None for missing Tf.data large row value


def calculate_ratio(a, b):
    return (b - a) / a


for i in range(1, len(frameworks)):
    print(f"{frameworks[i]}: {calculate_ratio(unlimited_memory[0], unlimited_memory[i])}")

for i in range(1, len(frameworks)):
    print(f"{frameworks[i]}: {calculate_ratio(limited_memory[0], limited_memory[i])}")

# Plotting
# Plotting
fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)  # Adjust figure width here
x = range(len(frameworks))  # x-axis positions

# Bar width
bar_width = 0.4

# Plotting small row, large row, and limited memory bars
ax.bar(x, unlimited_memory, width=bar_width, label="Unlim. Mem.", zorder=2)
ax.bar([p + bar_width for p in x], limited_memory, width=bar_width, label="Lim. Mem.", zorder=2)
theoretical_minimum = 153
ax.axhline(y=theoretical_minimum, color='g', linestyle='--', label='Theo. Min.', zorder=1)

# Adding text for zero value in large row
for i, value in enumerate(limited_memory):
    if value == 0:
        ax.text(i + 2 * bar_width, 10, "X", ha="center", va="bottom", color="red")

# Adding x-axis labels and title
# ax.set_xlabel('Systems')
ax.set_ylabel("Job completion time (s)")
# ax.set_title('Performance comparison of data processing frameworks')
ax.set_xticks([p + bar_width for p in x])
ax.set_xticklabels(frameworks)

# Adding grid
ax.grid(True, which="both", linestyle="--", zorder=0)

# Adding legend
ax.legend(loc="upper right")

# Saving the plot as a PDF
plt.tight_layout()  # Adjust layout

plt.savefig("synthetic.pdf", format="pdf")

# Showing the plot
# plt.show()
