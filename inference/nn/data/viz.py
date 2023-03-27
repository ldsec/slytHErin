import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the values of X
X = [3, 5, 10]

# Define the values of Y
Y = ['pt', 'ct']

# Initialize an empty dictionary to store the time values for each combination of X and Y
time_dict = {}

# Loop through each combination of X and Y
for x in X:
    for y in Y:
        # Get the filename of the csv file corresponding to the current combination of X and Y
        filename = f"nn20_batch292_parties{x}_model{y}_distrBtp.csv"

        # Load the csv file into a pandas dataframe
        df = pd.read_csv(filename)

        # Get the last row of the dataframe
        last_row = df.iloc[-1]

        # Get the time value from the last row
        time_value = last_row['Time']

        # Add the time value to the dictionary for the current combination of X and Y
        time_dict[(x, y)] = time_value / 1000.0

# Create a vertical bar plot for each value of X, with two bars side by side for the time of Y=pt and Y=ct
fig, ax = plt.subplots()
index = range(len(X))
bar_width = 0.35
opacity = 0.8
colors=["blue","red"]
for i, y in enumerate(Y):
    if y == "pt":
        label = "plaintext"
    else:
        label = "encrypted"
    time_values = [time_dict[(x, y)] for x in X]
    plt.bar([j + i*bar_width - bar_width/2 for j in index], time_values, bar_width,
            alpha=opacity,
            color=colors[i],
            label=label,
            zorder=0)
    plt.plot([j + i*bar_width - bar_width/2 for j in index], time_values, '-o', color=colors[i],zorder=0)
    # Add y values on the bars

for i, y in enumerate(Y):
    if y == "pt":
        label = "plaintext"
    else:
        label = "encrypted"
    time_values = [time_dict[(x, y)] for x in X]
    for j, v in enumerate(time_values):
            plt.text(j + i*bar_width - bar_width/2, v + 5, f"{v:.2f}", color='black', ha='center', zorder=(j+1)*500)

plt.xlabel('Parties')
plt.ylabel('Time(s)')
plt.xticks(index, X)
plt.legend()
plt.tight_layout()
plt.savefig("distr_btp.png")