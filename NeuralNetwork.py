import tensorflow as tf
import numpy as np

class LanderNeuralNetwork:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def train(self, X, y, epochs=100, verbose=1):
        self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def predict(self, X):
        return self.model.predict(X)

    def simulate_landing(self, fuel, velocity, height):
        initial_mass = 100.0  # Initial total mass (fuel mass + own mass)
        g = 9.8  # Acceleration due to gravity

        # Normalize the inputs
        fuel_norm = fuel / initial_mass
        velocity_norm = velocity / 10.0  # Normalizing velocity based on typical maximum landing velocity
        height_norm = height / 100.0  # Normalizing height based on typical maximum landing height

        # Create input data
        input_data = [[fuel_norm, velocity_norm, height_norm]]

        # Predict engine force
        engine_force_norm = self.predict(input_data)[0][0]

        # Calculate new mass based on fuel consumption
        fuel_consumed = fuel * engine_force_norm
        new_mass = initial_mass - fuel_consumed

        # Calculate gravitational force
        gravity_force = new_mass * g

        # Calculate engine force in real-world units
        engine_force = engine_force_norm * gravity_force

        return engine_force


# Create an instance of the LanderNeuralNetwork class
network = LanderNeuralNetwork()

# Generate random training data
num_samples = 1000

fuel = np.random.uniform(0, 100, num_samples)  # Random fuel values between 0 and 100
velocity = np.random.uniform(0, 20, num_samples)  # Random velocity values between 0 and 20
height = np.random.uniform(0, 500, num_samples)  # Random height values between 0 and 500

# Generate target labels based on a simple landing strategy
engine_force = 0.1 * fuel + 0.02 * velocity + 0.05 * height  # Arbitrary coefficients

# Normalize the input data
fuel_norm = fuel / 100.0
velocity_norm = velocity / 20.0
height_norm = height / 500.0

# Combine normalized inputs into a single array
X = np.column_stack((fuel_norm, velocity_norm, height_norm))

# Reshape the target labels to match the expected shape for model training
y = engine_force.reshape(-1, 1)


# Train the network using your training data (X) and target labels (y)
network.train(X, y)

# Simulate landing with specific fuel, velocity, and height values
fuel = 50.0
velocity = 8.0
height = 200.0

engine_force = network.simulate_landing(fuel, velocity, height)
print("Engine force:", engine_force)
