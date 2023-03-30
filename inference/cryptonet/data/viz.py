import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

font = {'size': 14}
font_axis = {'size': 14}
plt.rc('font', **font)
figsize = (6,4)
dpi = 1200
plt.tight_layout()
# Define the batch values we're interested in
batch_values = [1, 32, 64, 83, 256, 1024, 2048, 4096]

# Loop over the batch values
x_values = []
latency = []
y_errors = []
amort = []

optimal_batch_time = 0
file_name = f'cryptonet_batch{83}.csv'
if not os.path.exists(file_name):
    print(f'Error: file "{file_name}" not found')
    exit(1)

# Load the CSV file into a Pandas dataframe
df = pd.read_csv(file_name)

# Get the last row of the dataframe, which should contain the averages
last_row = df.tail(1)

# Get the time value from the last row
time = last_row['Time'].iloc[0] / 1000.0
optimal_batch_time = time

for x in batch_values:
    # Find the CSV file corresponding to this batch value
    file_name = f'cryptonet_batch{x}.csv'
    if not os.path.exists(file_name):
        print(f'Error: file "{file_name}" not found')
        continue

    # Load the CSV file into a Pandas dataframe
    df = pd.read_csv(file_name)

    # Get the last row of the dataframe, which should contain the averages
    last_row = df.tail(1)

    # Get the time value from the last row
    time = last_row['Time'].iloc[0] / 1000.0
    if x == 83:
        optimal_batch_time = time
        #continue

    # If the batch value in the CSV file doesn't match x, adjust the time value
    batch = last_row['Batch'].iloc[0]
    if batch != x and optimal_batch_time != 0:
        print(f"Adjusting {batch}->{x}")
        old = time
        time += (x // 83) * optimal_batch_time
        print(f"Time {old}->{time}")

    # Append the x and y values to the lists we'll use for plotting
    x_values.append(x)
    latency.append(time)
    amort.append(time/batch)
    y_errors.append(last_row['StdDev'].iloc[0] / 1000.0)

# Plot the results
fig1, ax1 = plt.subplots()
ax1.plot(x_values, latency, linewidth=1, color="blue", marker="o", zorder=0, label="Latency(s)")

for i, y in enumerate(latency):
    offset_x = 0
    offset_y = 3
    if i == 0:
        offset_y = 5
        offset_x = 0
    if i == 1:
        offset_x = -10
    if i == 2:
        offset_x = -10
        offset_y = 5
    if i == 3:
        offset_x = 50
        offset_y = -10
    if i == 4:
        offset_x = 200
        offset_y = -10
    if i == 5:
        offset_x = 1000
    if i == 6:
        offset_x = -1100
    if i == 7:
        offset_y = -10
        offset_x = -2000

    ax1.text(x_values[i]+offset_x, y+offset_y, f"{y:.2f}", color="blue", ha="center", zorder=1)

ax1.set_xscale("log")
ax1.set_xticks(x_values)
ax1.set_xticklabels([str(x) for x in x_values])
for i,tick in enumerate(ax1.get_xticklabels()):
    tick.set_rotation(60)
ax1.set_xlabel("Batch Size", font_axis)
ax1.set_ylabel("Latency(s)", font_axis)
fig1.savefig("latency.png", dpi=dpi,bbox_inches="tight")


fig2, ax2 = plt.subplots()
ax2.plot(x_values, amort, linewidth=1, color="red", marker="s", label="Amortized(s/sample)")
for i,y in enumerate(amort):
    offset_x = 0
    offset_y = .3
    if i == 0:
        offset_x = 0
    if i == 1:
        offset_y = .65
    if i == 2:
        offset_x = -8
    if i == 3:
        offset_x = 50
        offset_y = -.35
    if i == 6:
        offset_y = -.5
    if i == 7:
        offset_x = -2000
        offset_y = 0
    ax2.text(x_values[i]+offset_x, y+offset_y, f"{y:.2f}", color="red", ha="center", zorder=1)

ax2.set_xscale('log')
ax2.set_xticks(x_values)
ax2.set_xticklabels([str(x) for x in x_values])
for i,tick in enumerate(ax2.get_xticklabels()):
    tick.set_rotation(60)
ax2.set_ylabel('Amortized(s/sample)',font_axis)
ax2.set_xlabel('Batch Size',font_axis)
fig2.savefig("amortized.png", dpi=dpi,bbox_inches="tight")
