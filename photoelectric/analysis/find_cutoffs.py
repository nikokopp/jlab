import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file = "data/photoelectric_data_clean.csv"

data = pd.read_csv(data_file)

print(data.head())