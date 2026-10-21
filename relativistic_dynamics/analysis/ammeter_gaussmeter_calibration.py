import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress

amps = [0.0, 1.0, 2.0, 3.0, 3.8, 3.85, 4.0, 4.15, 4.1, 4.2, 4.3, 5.0]
gauss = [3.7, 26.2, 47.5, 70.0, 87.5, 89.4, 91.9, 94.6, 95.9, 96.8, 98.6, 114]

result = linregress(amps, gauss)

print(result)
slope = result[0]
intercept = result[1]
slope_err = result[4]

print(f"Slope = {slope:.3f} +/- {slope_err:.3f}")
print(f"Intercept = {intercept:.3f} +/- ")
