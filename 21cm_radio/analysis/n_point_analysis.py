import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import json

data_dir = "/home/rck67/jlab/21cm_radio/data/raw/10-19-2026/sun/SRT_SPEC_SAVE-2026_10_19_14_05_28.fits"

records = []
target = "Sun"

with fits.open(data_dir) as hdul:
    for i, hdu in enumerate(hdul):
        # just include Sun
        if hdu.header["OBJECT"].strip() != target:
            continue

        metadata = json.loads(hdu.header["METADATA"])

        az = metadata["motor_az"]
        el = metadata["motor_el"]

        # extract channels 0 and 1
        channel0 = hdu.data[0, 0, :, 0]
        channel1 = hdu.data[1, 1, :, 0]

        # for now, average across both channels & freq
        mean_signal = np.mean((channel0 + channel1) / 2)

        records.append({
            "hdu": i,
            "az": az,
            "el": el,
            "signal": mean_signal
        })

df = pd.DataFrame(records)
print(df.head())

plt.figure(figsize=(9, 5))

plt.plot(df["hdu"], df["signal"], ".-")

plt.xlabel("HDU index")
plt.ylabel("Mean measured temperature (K)")
plt.title("Sun N-Point scan Signal")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()