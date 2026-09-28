import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# naming variables
data_file = "uranium.spe"
figure_file = "figures/uranium_histogram.png"
num_bins = 200

# load data
counts = np.loadtxt(data_file, skiprows=12, max_rows=2048)
bins = np.arange(len(counts))

# plot histogram
plt.style.use("dark_background")
plt.hist(bins, bins=num_bins, weights=counts, color="lightsteelblue")

plt.xlabel("Bin Number")
plt.ylabel("Counts")
plt.title("Natural Uranium Spectrum")

plt.tight_layout()
plt.savefig(figure_file)
plt.show()

