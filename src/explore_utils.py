import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Jika butuh visualisasi di .py
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(self):
    print("\n--- Data Overview ---")
    print(self.df.head())
    print("\n--- Info ---")
    print(self.df.info())
    print("\n--- Descriptive Statistics ---")
    print(self.df.describe())
    # Tambahan: cek missing values, korelasi, outlier
    print("\n--- Missing Values ---")
    print(self.df.isnull().sum())
    print("\n--- Correlation Matrix ---")
    print(self.df.corr())
