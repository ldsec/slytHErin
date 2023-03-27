import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the batch values we're interested in
batch_values = [1, 32, 64, 83, 256, 1024, 2048, 4096]

# Loop over the batch values
x_values = []
y_values = []
y_errors = []

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

    # If the batch value in the CSV file doesn't match x, adjust the time value
    batch = last_row['Batch'].iloc[0]
    if batch != x and optimal_batch_time != 0:
        print(f"Adjusting {batch}->{x}")
        old = time
        time += (x // 83) * optimal_batch_time
        print(f"Time {old}->{time}")

    # Append the x and y values to the lists we'll use for plotting
    x_values.append(x)
    y_values.append(time)
    y_errors.append(last_row['StdDev'].iloc[0] / 1000.0)

# Plot the results
#plt.errorbar(x_values, y_values, yerr=y_errors, elinewidth=2, fmt='o', color="blue")
plt.plot(x_values, y_values, linewidth=2, color="green", marker="o", linestyle="-.", zorder=0)
for i,y in enumerate(y_values):
    j = 1
    if i == 2:
        j = 0.5
    plt.text(x_values[i], y + 2*j, f"{y:.2f}", color="black", ha="center", zorder=1)
plt.xticks(x_values)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Batch")
plt.ylabel("Time(s)")
plt.savefig("flexible_batch.png")