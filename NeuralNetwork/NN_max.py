import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize parameters with He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return Z > 0

    def forward(self, X):
        # Forward propagation
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2  # Linear activation for output (regression)

    def compute_loss(self, y, y_hat):
        # Mean Squared Error
        return np.mean((y - y_hat) ** 2)

    def backward(self, X, y, y_hat, learning_rate):
        # Backward propagation
        m = X.shape[0]

        # Output layer gradients
        dZ2 = (y_hat - y) / m
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            y_hat = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y, y_hat)
            losses.append(loss)

            # Backward pass
            self.backward(X, y, y_hat, learning_rate)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        return self.forward(X)


# Load and prepare data
d = pd.read_csv("../data/weather_daily_2020-03-26_to_2025-05-24.csv")
x = d[['Min Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)', 'Precipitation (mm)', 'Wind Speed (km/h)']]
y = d['Max Temperature (°C)'].values.reshape(-1, 1)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert to numpy arrays
X_train = np.array(x_train_scaled)
y_train = np.array(y_train)
X_test = np.array(x_test_scaled)
y_test = np.array(y_test)

# Initialize and train neural network
input_size = X_train.shape[1]
nn = NeuralNetwork(input_size=input_size, hidden_size=64, output_size=1)
loss_history = nn.train(X_train, y_train, epochs=10000, learning_rate=0.02)

# Evaluate
train_pred = nn.predict(X_train)
test_pred = nn.predict(X_test)

print(f"\nTraining R²: {r2_score(y_train, train_pred):.4f}")
print(f"Test R²: {r2_score(y_test, test_pred):.4f}")

# Save parameters and scaler
np.savez('max_temp_nn_params.npz',
         W1=nn.W1, b1=nn.b1,
         W2=nn.W2, b2=nn.b2)
np.save('max_temp_scaler_mean.npy', scaler.mean_)
np.save('max_temp_scaler_scale.npy', scaler.scale_)