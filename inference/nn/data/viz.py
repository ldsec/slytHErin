import matplotlib.pyplot as plt
import pandas as pd

X = [3, 5, 10, 20]

time_values = []

for x in X:
    # Get the filename of the csv file corresponding to the current combination of X and Y
    filename = f"nn20_batch292_parties{x}_modelct_distrBtp.csv"

    # Load the csv file into a pandas dataframe
    df = pd.read_csv(filename)

    # Get the last row of the dataframe
    last_row = df.iloc[-1]

    # Get the time value from the last row
    time_value = last_row['Time']
    batch = last_row['Batch']

    time_values.append(time_value / (1000.0 * batch))

# Get the centralized time values
df = pd.read_csv("nn20_batch585_modelct_datapt.csv")
last_row = df.iloc[-1]
time_value = last_row['Time']
batch = last_row['Batch']
centralized_time = [time_value / (1000.0 * batch) for _ in X]

# Create a vertical bar plot for each value of X, with two bars side by side for the time of Y=pt and Y=ct
fig, ax = plt.subplots()
index = range(len(X))
bar_width = 0.35
opacity = 0.8
colors=["blue","red"]

plt.bar([j - bar_width/2 for j in index], time_values, bar_width,
        alpha=opacity,
        color="blue",
        label="distributed",
        zorder=0)
plt.bar([j + bar_width/2 for j in index], centralized_time, bar_width, color="red", label="centralized", zorder=0)

# Plot a line connecting the values in time_values
plt.plot([j - bar_width/2 for j in index], time_values, '-o', color="blue", zorder=0)

# Add labels to the bars
for j, v in enumerate(time_values):
    plt.text(j - bar_width/2, v + 5e-2, f"{v:.2f}", color='black', ha='center', zorder=(j+1)*500)
for j, v in enumerate(centralized_time):
    plt.text(j + bar_width - bar_width/2, v + 1e-2, f"{v:.2f}", color='black', ha='center', zorder=(j+1)*500)

# Add axis labels and ticks
plt.xlabel('Parties')
plt.ylabel('Amortized Time(s/sample)')
plt.xticks(index, X)
plt.legend()

# Save the plot to a file
plt.savefig("distr_vs_centralized.png")
