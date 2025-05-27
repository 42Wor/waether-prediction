import numpy as np
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

# Load parameters
params = np.load('max_temp_nn_params.npz')
scaler_mean = np.load('max_temp_scaler_mean.npy')
scaler_scale = np.load('max_temp_scaler_scale.npy')

# Create new network instance
nn = NeuralNetwork(input_size=5, hidden_size=64, output_size=1)
nn.W1 = params['W1']
nn.b1 = params['b1']
nn.W2 = params['W2']
nn.b2 = params['b2']

# Prepare new data (example)
new_data = np.array([[15.0, 65.0, 1012.0, 2.0, 20.0]])
scaled_data = (new_data - scaler_mean) / scaler_scale

# Make prediction
prediction = nn.predict(scaled_data)
print(f"Predicted max temperature: {prediction[0][0]:.1f}°C")

# Example predictions using trained model
test_samples = [
    ([15.0, 65.0, 1012.0, 2.0, 20.0], "T1"),
    ([20.0, 60.0, 1010.0, 5.0, 25.0], "T2"),
    ([25.0, 55.0, 1008.0, 10.0, 30.0], "T3"),
    ([18.0, 70.0, 1011.0, 0.0, 12.0], "T4"),
]



for features, label in test_samples:
    features_arr = np.array([features])
    scaled_features = (features_arr - scaler_mean) / scaler_scale
    predicted_T_Max = nn.predict(scaled_features)
    print(f"{label} Predicted Max Temperature (°C): {predicted_T_Max[0][0]:.1f}")



