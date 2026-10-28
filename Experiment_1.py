import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())
print(df.describe())

# Select single feature
X = df[['AveRooms']].values
print(len(X))
y = df['MedHouseVal'].values

# Clip outliers (optional, for better visualization)
mask = (X.flatten() < 10) & (y < 5)
X, y = X[mask], y[mask]

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Gradient Descent
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m = len(y)
    theta0, theta1 = 0.0, 0.0
    cost_history = []

    for _ in range(epochs):
        y_p = theta0 + theta1 * X.flatten()
        error = y_p - y

        theta0 -= lr * (1/m) * np.sum(error)
        theta1 -= lr * (1/m) * np.sum(error * X.flatten())

        cost_history.append((1/(2*m)) * np.sum(error**2))

    return theta0, theta1, cost_history

t0_gd, t1_gd, costs = gradient_descent(
    X_train_sc, y_train, lr=0.01, epochs=1000
)

print(f'GD => theta0={t0_gd:.4f}, theta1={t1_gd:.4f}')

# Plot cost convergence
plt.plot(costs)
plt.title('Cost Function Convergence (Gradient Descent)')
plt.xlabel('Iterations')
plt.ylabel('MSE Cost')
plt.show()

# Normal Equation
def normal_equation(X, y):
    X_b = np.c_[np.ones(len(X)), X]
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return theta[0], theta[1]

t0_ne, t1_ne = normal_equation(X_train_sc, y_train)

print(f'NE => theta0={t0_ne:.4f}, theta1={t1_ne:.4f}')

# Evaluation
for name, t0, t1 in [
    ('Gradient Descent', t0_gd, t1_gd),
    ('Normal Equation', t0_ne, t1_ne)
]:
    y_pred = t0 + t1 * X_test_sc.flatten()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{name:20s} MSE={mse:.4f} RMSE={mse**0.5:.4f} R2={r2:.4f}')

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (name, t0, t1) in zip(axes, [
    ('Gradient Descent', t0_gd, t1_gd),
    ('Normal Equation', t0_ne, t1_ne)
]):
    ax.scatter(
        X_test_sc.flatten(), y_test,
        alpha=0.4, s=8, color='steelblue', label='Data'
    )

    xs = np.linspace(X_test_sc.min(), X_test_sc.max(), 200)
    ax.plot(
        xs, t0 + t1 * xs,
        color='crimson', lw=2, label='Fitted Line'
    )

    ax.set_title(name)
    ax.set_xlabel('AveRooms (Scaled)')
    ax.set_ylabel('Median House Value')
    ax.legend()

plt.suptitle('Linear Regression: Gradient Descent vs Normal Equation')
plt.tight_layout()
plt.show()
