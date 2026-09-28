import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data_file = "pendulumData.csv"
figure_file = "figures/pendulum_histogram.png"

# 1. Load data
data = pd.read_csv(data_file)
g = data["g (m/s2)"]

# 2. Inspect data and get rid of outliers
g_mean = g.mean()
g_stdv = g.std()

print("g mean:", g_mean)
print("g stdv:", g_stdv)

# remove data outside of 4 standard deviations
g_filtered = g.where(g < (g_mean + 3*g_stdv))

# # plot the data to inspect
# plt.style.use("dark_background")
# fig, ax = plt.subplots()

# ax.scatter(np.arange(len(g)), g_filtered, s=15)
# ax.set_xlabel("Measurement number")
# ax.set_ylabel(r"$g$ (m/s$^2$)")

# 3. Bin data into histogram
counts, bin_edges = np.histogram(g_filtered.dropna(), bins=30)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# 4. Fit to Gaussian
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def chi_squared(params, x, data):
    A, mu, sigma = params

    model = gaussian(x, A, mu, sigma)

    # Prevent division by an absurdly tiny model value
    model = np.maximum(model, 1e-12)

    chi2 = np.sum((data - model)**2 / model)

    return chi2

initial_guess = [np.max(counts), np.mean(g_filtered),  np.std(g_filtered)]
result = minimize(chi_squared, initial_guess, args=(bin_centers, counts))

A_fit, mu_fit, sigma_fit = result.x

print("Converged:", result.success)
print("Message:", result.message)

print("Amplitude:", A_fit)
print("Mean:", mu_fit)
print("Sigma:", sigma_fit)

print("Minimum chi squared:", result.fun)

plt.style.use("dark_background")

plt.errorbar(
    bin_centers,
    counts,
    yerr=np.sqrt(counts),
    fmt='o',
    label='Data'
)

x_smooth = np.linspace(bin_edges[0], bin_edges[-1], 500)

plt.plot(
    x_smooth,
    gaussian(x_smooth, A_fit, mu_fit, sigma_fit),
    label='Gaussian fit'
)

plt.xlabel(r"$g$ (m/s$^2$)")
plt.ylabel("Counts per bin")
plt.legend()

plt.show()