import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the batch values we're interested in
batch_values = [1, 256, 1024, 2048, 4096]

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
        continue

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
    y_errors.append(last_row['StdDev'].iloc[0] / 1000.0)

# Plot the results
plt.plot(x_values, latency, linewidth=2, color="green", marker="o", zorder=0)


plt.xticks(x_values)
plt.yticks(latency)
plt.xscale("log")
#plt.yscale("log")
plt.xlabel("Batch")
plt.ylabel("Latency(s)")
plt.savefig("flexible_batch.png")