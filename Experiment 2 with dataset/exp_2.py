# Step 1: Import Libraries and Load Dataset

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Folder this script lives in, so it works no matter where you launch it from
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def find_data(filename):
    """Find a data file next to this script or in any folder below it."""
    match = next(HERE.rglob(filename), None)
    if match is None:
        raise FileNotFoundError(f"Could not find '{filename}' in or below {HERE}")
    return match


# Load Auto MPG dataset (local CSV next to this script)
df = pd.read_csv(find_data('auto-mpg.csv'))
print(df.head())
print(df.describe())

# Step 2: Exploratory Data Analysis

# Examine the target variable distribution
plt.figure(figsize=(8, 4))
plt.hist(df['mpg'], bins=30, color='steelblue', edgecolor='white')
plt.title('Distribution of MPG')
plt.xlabel('Miles Per Gallon (MPG)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of chosen feature vs target
plt.scatter(df['displacement'], df['mpg'], alpha=0.5, s=15, color='steelblue')
plt.xlabel('Engine Displacement (cu. in.)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.title('Displacement vs MPG')
plt.show()

# Step 3: Data Preprocessing

# Drop rows with missing values
df = df.dropna(subset=['displacement', 'mpg'])
# Select single feature and target
X = df[['displacement']].values
y = df['mpg'].values
# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=4)

# Step 4: Baseline — Simple Linear Regression

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print(f'Linear Regression => MSE={mse_lin:.4f} RMSE={mse_lin**0.5:.4f} R2={r2_lin:.4f}')

# Step 5: Polynomial Regression of Varying Degrees

degrees = [1, 2, 3, 4, 6, 8]
results = []
models = {}
for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((d, mse, mse**0.5, r2))
    models[d] = (poly, model)
    print(f'Degree {d:2d} => MSE={mse:.4f} RMSE={mse**0.5:.4f} R2={r2:.4f}')

# Step 6: Compare Polynomial Models with Linear Regression

results_df = pd.DataFrame(results, columns=['Degree', 'MSE', 'RMSE', 'R2'])
results_df.loc[-1] = ['Linear (deg=1 baseline)', mse_lin, mse_lin**0.5, r2_lin]
results_df = results_df.sort_index().reset_index(drop=True)
print(results_df)
plt.figure(figsize=(7, 4))
plt.plot(degrees, [r[3] for r in results], marker='o', color='crimson')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score (Test Set)')
plt.title('Model Performance vs Polynomial Degree')
plt.grid(alpha=0.3)
plt.show()

# Step 7: Visualization — Fitted Curves

xs = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
plt.figure(figsize=(9, 6))
plt.scatter(X_test, y_test, alpha=0.4, s=15, color='steelblue', label='Test Data')
# Linear baseline
plt.plot(xs, lin_model.predict(xs), color='black', lw=2, linestyle='--', label='Linear (deg=1)')
# A few representative polynomial degrees
for d, color in zip([2, 4, 8], ['crimson', 'green', 'orange']):
    poly, model = models[d]
    xs_poly = poly.transform(xs)
    plt.plot(xs, model.predict(xs_poly), color=color, lw=2, label=f'Degree {d}')
plt.xlabel('Engine Displacement (cu. in.)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.title('Polynomial Regression Fits vs Linear Baseline')
plt.legend()
plt.tight_layout()
plt.show()
