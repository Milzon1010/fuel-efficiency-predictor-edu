import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

class FuelEfficiencyPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)

    def prepare_data(self, feature_cols, target_col):
        X = self.df[feature_cols]
        y = self.df[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train, use_rf=False):
        if use_rf:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def compare_models(self, X_train, y_train, X_test, y_test):
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            print(f"{name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
        return results
