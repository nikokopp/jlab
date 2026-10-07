import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.optimize import curve_fit

data_file = Path("/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv")
reverse_fit_file = Path("/home/rck67/jlab/photoelectric/data/reverse_current_fits.csv")
