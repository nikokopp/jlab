import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

data_dir = "/home/rck67/jlab/21cm_radio/data/raw/10-19-2026/sun/SRT_SPEC_SAVE-2026_10_19_14_05_28.fits"

sun_data = fits.open(data_dir)

# plot one integration
hdu = sun_data[100]

data = hdu.data
header = hdu.header

channel0 = data[0, 0, :, 0]
channel1 = data[1, 1, :, 0]

n_bins = len(channel0)
pixels = np.arange(n_bins)

freq_hz = (header["CRVAL2"] + (pixels + 1 - header["CRPIX2"]) * header["CDELT2"])
freq_mhz = freq_hz / 1e6

plt.figure(figsize=(9, 5))

plt.plot(freq_mhz, channel0, color="#4d45c4", label="Channel 0")
plt.plot(freq_mhz, channel1, color="#c65097", label="Channel 1")

plt.xlabel("Frequency (Mhz)")
plt.ylabel("Measured temp (K)")
plt.title("Sun radio spectrum (integration 100)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()