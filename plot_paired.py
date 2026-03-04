from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# locate paired files in data/paired directory
paired_dir = Path(__file__).parent / 'data' / 'paired'
stream_files = list(paired_dir.glob('stream-gages_*_synced_smoothed.csv'))
if not stream_files:
    raise RuntimeError("No synced stream file found in data/paired")
stream_file = stream_files[0]

rain_files = list(paired_dir.glob('rain-gages_*_synced_smoothed.csv'))
if not rain_files:
    raise RuntimeError("No synced rain files found in data/paired")

# read data
stream_df = pd.read_csv(stream_file, parse_dates=["Date"])
rain_dfs = {rf: pd.read_csv(rf, parse_dates=["Date"]) for rf in rain_files}

# ensure timestamps align
for rf, df in rain_dfs.items():
    if not stream_df['Date'].equals(df['Date']):
        print(f"Warning: timestamps in {rf.name} do not exactly match stream. Plotting by index.")

# create figure with two subplots sharing x-axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# plot stream stage on left y-axis
color = 'tab:blue'
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Stage', color=color)
ax1.plot(stream_df['Date'], stream_df['Stage'], color=color, label='Stage')
ax1.tick_params(axis='y', labelcolor=color)

# create second y-axis for rain
ax2 = ax1.twinx()
ax2.set_ylabel('Rain Amount (in)', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

# plot each rain file
colors = ['tab:green', 'tab:olive', 'tab:cyan', 'tab:magenta']
for idx, (rf, df) in enumerate(rain_dfs.items()):
    clr = colors[idx % len(colors)]
    label = rf.name.replace('.csv','')
    ax2.plot(df['Date'], df['Rain Amount (in)'], color=clr, label=label)

# title and layout
plt.title('Stream Stage and Rain Amount vs Time')
fig.tight_layout()

# legend: combine handles from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

# format x-axis dates
from matplotlib.dates import DateFormatter, AutoDateLocator
locator = AutoDateLocator()
formatter = DateFormatter('%Y-%m-%d %H:%M')
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
fig.autofmt_xdate()

# add grid
ax1.grid(True, which='major', linestyle='--', alpha=0.5)
ax2.grid(False)

# save figure to file
output_plot = paired_dir.parent / "paired_plot.png"
plt.savefig(output_plot, dpi=150)
print(f"Plot saved to {output_plot}")

# show plot
#plt.show()
plt.close()

# --- rain-only plot --------------------------------------------------------
print("Creating rain-only plot...")
fig2, axr = plt.subplots(figsize=(12, 6))
for idx, (rf, df) in enumerate(rain_dfs.items()):
    clr = colors[idx % len(colors)]
    label = rf.name.replace('.csv','')
    axr.plot(df['Date'], df['Rain Amount (in)'], color=clr, label=label)

axr.set_xlabel('Timestamp')
axr.set_ylabel('Rain Amount (in)')
plt.title('Rain Amount vs Time')
fig2.tight_layout()

# legend
axr.legend(loc='upper left')

# format x-axis same as before
axr.xaxis.set_major_locator(locator)
axr.xaxis.set_major_formatter(formatter)
fig2.autofmt_xdate()

# grid
axr.grid(True, which='major', linestyle='--', alpha=0.5)

# save rain-only figure
rain_plot = paired_dir.parent / "rain_only_plot.png"
plt.savefig(rain_plot, dpi=150)
print(f"Rain-only plot saved to {rain_plot}")
plt.close()

# --- individual rain plots -------------------------------------------------
for rf, df in rain_dfs.items():
    print(f"Creating individual plot for {rf.name}...")
    figr, ax = plt.subplots(figsize=(12, 6))
    clr = colors[list(rain_dfs.keys()).index(rf) % len(colors)]
    ax.plot(df['Date'], df['Rain Amount (in)'], color=clr, label=rf.name)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Rain Amount (in)')
    plt.title(rf.name + ' Rain Amount vs Time')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    figr.autofmt_xdate()
    ax.grid(True, which='major', linestyle='--', alpha=0.5)

    single_plot = paired_dir.parent / f"{rf.stem}_plot.png"
    plt.savefig(single_plot, dpi=150)
    print(f"Individual rain plot saved to {single_plot}")
    #plt.show()
    plt.close()
