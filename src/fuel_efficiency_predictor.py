import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class FuelEfficiencyPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path, na_values='?')
        self.df.dropna(inplace=True)
        print(f"Data loaded! Shape: {self.df.shape}")

    def explore_data(self):
        print("\n--- Data Overview ---")
        print(self.df.head())
        print("\n--- Info ---")
        print(self.df.info())
        print("\n--- Descriptive Statistics ---")
        print(self.df.describe())

    def train_model(self, feature_cols, target_col):
        X = self.df[feature_cols]
        y = self.df[target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        print("Model trained!")

    def evaluate_model(self):
        if self.model is None:
            print("Model belum dilatih.")
            return
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print("\n--- Evaluation Results ---")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"R2 Score: {r2:.3f}")
        print("\n--- Model Coefficients ---")
        for name, coef in zip(self.X_train.columns, self.model.coef_):
            print(f"{name}: {coef:.3f}")
        print(f"Intercept: {self.model.intercept_:.3f}")
