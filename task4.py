import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

# Generate data
np.random.seed(0)
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

# 1.1.a Linear Regression
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_test)

# 1.1.b Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
x_poly_train = poly_features.fit_transform(x_train)
x_poly_test = poly_features.transform(x_test)

poly_model = LinearRegression()
poly_model.fit(x_poly_train, y_train)
y_pred_poly = poly_model.predict(x_poly_test)

# 1.1.c Neural Network
nn_model = MLPRegressor(hidden_layer_sizes=(6,), max_iter=1000, random_state=0)
nn_model.fit(x_train, y_train.ravel())
y_pred_nn = nn_model.predict(x_test)

# 1.1.d Calculate MSE
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_poly = mean_squared_error(y_test, y_pred_poly)
mse_nn = mean_squared_error(y_test, y_pred_nn)

print(f"Mean Squared Error (Linear): {mse_linear}")
print(f"Mean Squared Error (Polynomial): {mse_poly}")
print(f"Mean Squared Error (Neural Network): {mse_nn}")

# 1.1.e Plot results
plt.figure(figsize=(12, 6))

# Data points
plt.scatter(x_data, y_data, color='blue', alpha=0.5, label='All data points')
plt.scatter(x_train, y_train, color='green', alpha=0.5, label='Training data')
plt.scatter(x_test, y_test, color='red', alpha=0.5, label='Testing data')

# Linear Regression
plt.plot(x_test, y_pred_linear, color='red', label='Linear Regression')

# Polynomial Regression
x_plot = np.linspace(-0.5, 0.5, 100)[:, np.newaxis]
y_plot = poly_model.predict(poly_features.transform(x_plot))
plt.plot(x_plot, y_plot, color='green', label='Polynomial Regression')

# Neural Network
y_plot_nn = nn_model.predict(x_plot)
plt.plot(x_plot, y_plot_nn, color='orange', label='Neural Network')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Data Points and Regression Models')
plt.legend()
plt.show()

# Print some sample data points
print("\nSample Training Data:")
for i in range(5):
    print(f"x: {x_train[i][0]:.4f}, y: {y_train[i][0]:.4f}")

print("\nSample Testing Data:")
for i in range(5):
    print(f"x: {x_test[i][0]:.4f}, y: {y_test[i][0]:.4f}")