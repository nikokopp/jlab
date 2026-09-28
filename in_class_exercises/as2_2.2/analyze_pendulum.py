import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file = "pendulumData.csv"
figure_file = "figures/pendulum_histogram.png"
bins = 10

# load data
data = pd.read_csv(data_file)
g = data["g (m/s2)"].to_numpy()

# plot the data to inspect
fig, ax = plt.subplots()

ax.scatter(np.arange(len(g)), g, s=15)

ax.set_xlabel("Measurement number")
ax.set_ylabel(r"$g$ (m/s$^2$)")

fig.tight_layout()
plt.show()

# going to remove the 978 point and 0.6 point
g = 


# statistical analysis
print("average g:", g.mean())
print("stdev g:", pd.std(g))
print("num measurements:", len(g))