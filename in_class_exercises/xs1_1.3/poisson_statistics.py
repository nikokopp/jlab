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

plt.errorbar(centers, counts, 
             yerr=np.sqrt(counts), 
             fmt="none", 
             ecolor="red", 
             capsize=2)

plt.xlabel("Time (ms)")
plt.ylabel("Counts per 10s interval")
plt.title("Coincidence Counts")

plt.tight_layout()
plt.savefig("coincidences_histogram.png")

# cumulative average
j = np.arange(1, len(counts) + 1)
mu_c = []
for i in j:
    mu_c.append(sum(counts[0:i])/i)

plt.figure()
plt.scatter(j, mu_c, color="lightsteelblue")
plt.xlabel("Number of 10s intervals")
plt.ylabel("Cumulative mean")

mu_c = np.array(mu_c)
mu_c_err = np.sqrt(mu_c/j)

plt.errorbar(j, mu_c, 
             yerr=mu_c_err, 
             fmt="none", 
             ecolor="lightgray", 
             elinewidth=0.7,
             capsize=1.5, capthick=0.7
             )
plt.tight_layout()
plt.savefig("cumulative_mean.png")

# print(f"Mean after 100 intervals: {mu_c[-1]}")
# print(f"Error on mean: {mu_c_err[-1]}")

# second calculation
bins2 = np.arange(1_000_000, 2_000_001, 10_000)
counts2, edges2, _ = plt.hist(times, bins2)

j2 = np.arange(1, len(counts2) + 1)

mu_c2 = []
for i in j2:
    mu_c2.append(sum(counts2[0:i]) / i)

mu_c2 = np.array(mu_c2)
mu_c2_err = np.sqrt(mu_c2/j2)

# plot the different means
plt.figure()
plt.errorbar(j, mu_c, yerr=mu_c_err, fmt=".", capsize=1.5, label="First 100 intervals")
plt.errorbar(j2, mu_c2, yerr=mu_c2_err, fmt=".", capsize=1.5, label="Second 100 intervals")
plt.legend()
plt.tight_layout()
plt.savefig("running_mean_comparison.png")

mean1 = mu_c[-1]
mean2 = mu_c2[-1]
error1 = mu_c_err[-1]
error2 = mu_c2_err[-1]

diff = abs(mean1 - mean2)
combined_err = np.sqrt(error1**2 + error2**2)

# print(f"First mean: {mean1})
# print(f"First error: {error1})
# print(f"Second mean: {mean2} 
# print(f"Second error: {error2})
# print(f"Difference: {diff}")
# print(f"Combined standard error: {combined_err}")
# print(f"Means differ by {diff / diff_err} combined standard errors.")

