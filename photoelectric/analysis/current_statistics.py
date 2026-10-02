import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_file = "/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv"

data = pd.read_csv(data_file)

wavelengths = [546.1, 435.7, 365]

# folder to save plots
figure_dir = Path("/home/rck67/jlab/photoelectric/figures/current_statistics")
figure_dir.mkdir(parents=True, exist_ok=True)

