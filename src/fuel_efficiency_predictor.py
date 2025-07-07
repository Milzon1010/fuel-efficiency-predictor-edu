import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class FuelEfficiencyPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None      # DataFrame untuk data
        self.model = None   # Model ML (Linear Regression)
        self.feature_cols = None

    def load_data(self):
        # Baca file, handle missing value pada horsepower (sering ada '?')
        self.df = pd.read_csv(self.data_path, na_values='?')
        self.df = self.df.dropna()
        print(f"Data loaded! Shape: {self.df.shape}")

    def explore_data(self):
        print("\n--- Data Overview ---")
        print(self.df.head())
        print("\n--- Info ---")
        print(self.df.info())
        print("\n--- Descriptive Statistics ---")
        print(self.df.describe())

    def prepare_data(self, feature_cols, target_col):
        self.feature_cols = feature_cols
        X = self.df[feature_cols]
        y = self.df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.model = model
        print("Model trained!")

    def predict(self, X):
        if self.model is not None:
            return self.model.predict(X)
        else:
            print("Model belum di-train.")

    def evaluate_model(self, X_test, y_test):
        if self.model is not None:
            predictions = self.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            print("\n--- Evaluation Results ---")
            print(f"Mean Squared Error (MSE): {mse:.3f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
            print(f"Mean Absolute Error (MAE): {mae:.3f}")
            print(f"R2 Score: {r2:.3f}")
            return mse, rmse, mae, r2
        else:
            print("Model belum di-train.")

    def print_coefficients(self):
        if self.model is not None and self.feature_cols is not None:
            print("\n--- Model Coefficients ---")
            for name, coef in zip(self.feature_cols, self.model.coef_):
                print(f"{name}: {coef:.3f}")
            print(f"Intercept: {self.model.intercept_:.3f}")
        else:
            print("Model or feature names not set.")

    def run(self, feature_cols, target_col):
        self.load_data()
        self.explore_data()
        X_train, X_test, y_train, y_test = self.prepare_data(feature_cols, target_col)
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        self.print_coefficients()
        return self.model

# --- Example Usage (run as script) ---
if __name__ == "__main__":
    data_path = 'data/auto-mpg.csv'
    feature_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
    target_col = 'mpg'

    predictor = FuelEfficiencyPredictor(data_path)
    model = predictor.run(feature_cols, target_col)
