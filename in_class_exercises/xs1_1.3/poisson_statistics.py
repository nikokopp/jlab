import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# name variables
data_file = 'FileC021.txt'

# load data
# keep just time in ms, dead time, and coincidences
df = pd.read_csv(
    data_file, 
    sep=r'\s+', 
    skiprows=7, 
    usecols=[3, 9, 10], 
    header=None,
    names=['Time (ms)', 'Dead Time (us)', 'Coincidence']
    )

# keep just the coincidences
df_coincidences = df[df['Coincidence'] == 1] # get coincidences from column 3

# graphing data
times = df_coincidences['Time (ms)']
bins = np.arange(0, 1_000_001, 10_000)

# plot histogram
plt.style.use("dark_background")
counts, edges, _ = plt.hist(times, bins, color="lightsteelblue")
centers = (edges[:-1] + edges[1:])/2 # error bars placed at centers

plt.errorbar(centers, counts, yerr=np.sqrt(counts), fmt="none", ecolor="white", capsize=3)

plt.xlabel("Time (ms)")
plt.ylabel("Counts per 10s interval")
plt.title("Coincidence Counts")

plt.tight_layout()
plt.savefig("coincidences_histogram.png")