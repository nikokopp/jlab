import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import odr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.special import erfc

from pathlib import Path
import re

data_root = Path("/home/rck67/jlab/relativistic_dynamics/data/raw/calibration")
data_file = Path(data_root)
output_dir = Path("/home/rck67/jlab/relativistic_dynamics/data/processed/calibration_results")

first_channel = 0

# visually determine what features to identify in spectrum
import numpy as np

def compton_edge(E_gamma):
    """
    Calculate Compton edge from gammay ray energy.
    """
    m_e_c2 = 510.99895  # electron rest energy in keV
    E_gamma = np.asarray(E_gamma)
    E_c = E_gamma / (1 + m_e_c2/(2 * E_gamma))
    return E_c

gamma_energies = [30.85, 35.10, 53.16, 79.62, 81.00, 160.61, 223.25, 276.40, 302.85, 356.02, 383.85]
compton_edges = compton_edge(gamma_energies)

for i in compton_edges:
    print(f"{i:.2f} keV")