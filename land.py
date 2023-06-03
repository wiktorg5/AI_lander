import numpy as np
import tensorflow as tf
import pickle


class LanderModel:

    def __init__(self, g, t, ml, tmax, mp0, Fmax):
        self.g = g
        self.t = t
        self.ml = ml
        self.tmax = tmax
        self.mp0 = mp0
        self.Fmax = Fmax
        self.k = mp0 / (Fmax * tmax)
        self.ko = 1 / self.k

    def simulate(self, state, F):
        g = self.g
        t = self.t
        ml = self.ml
        k = self.k
        ko = self.ko
        Fmax = self.Fmax

        mp = state[0]
        x = state[1]
        v = state[2]

        mc = ml + mp  # own mass + fuel mass
        F = np.clip(F, 0, Fmax)

        mp1 = mp - k * F * t
        if mp1 < 0:
            F = mp / (k * t)
            mp1 = 0

        if F > 0.1:
            v1 = ko * np.log(mc / (mc - k * F * t)) + v - g * t
            x1 = -(ko * ko / F) * np.log(mc / (mc - k * F * t)) * (mc - k * F * t) + t / k + v * t - (g * t * t) / 2 + x
        else:
            v1 = v - g * t
            x1 = v * t - (g * t * t) / 2 + x

        if x1 < 0:
            x1 = 0
            i = 1
            tpom = np.linspace(0, t, num=int(np.floor(t * 100)))
            if F > 0.1:
                while -(ko * ko / F) * np.log(mc / (mc - k * F * tpom[i])) * (mc - k * F * tpom[i]) + tpom[i] / k + v * \
                        tpom[i] - (g * tpom[i] * tpom[i]) / 2 + x > 0:
                    i = i + 1
                v1 = ko * np.log(mc / (mc - k * F * tpom[i])) + v - g * tpom[i]
            else:
                while v * tpom[i] - (g * tpom[i] * tpom[i]) / 2 + x > 0:
                    i = i + 1
                v1 = v - g * tpom[i]

        state_next = [mp1, x1, v1]
        return state_next


class NeuralController:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights = {
            'W1': tf.Variable(tf.random.normal([self.input_dim, self.hidden_dim], dtype=tf.float64)),
            'W2': tf.Variable(tf.random.normal([self.hidden_dim, self.output_dim], dtype=tf.float64))
        }

    def initialize_weights(self):
        weights = {}
        weights['W1'] = tf.Variable(tf.random.normal([self.input_dim, self.hidden_dim]))
        weights['W2'] = tf.Variable(tf.random.normal([self.hidden_dim, self.output_dim]))
        return weights

    def forward(self, x):
        hidden_layer = tf.nn.relu(tf.matmul(x, tf.cast(self.weights['W1'], tf.float64)))
        output_layer = tf.matmul(hidden_layer, tf.cast(self.weights['W2'], tf.float64))
        return output_layer


# Set up the lander model and parameters
g = 8.15
t = 1
ml = 700
tmax = 200
mp0 = 1000
h0 = 1000
v0 = 0
Fmax = 15000
lander = LanderModel(g, t, ml, tmax, mp0, Fmax)

# Set up the genetic algorithm parameters
population_size = 100
input_dim = 3
hidden_dim = 9
output_dim = 1
mutation_rate = 0.01
generations = 20
file_path = './saved_controller/best_controller_generation_824.pkl'  # Replace X with the desired generation number
with open(file_path, 'rb') as f:
    best_weights = pickle.load(f)

best_controller = NeuralController(input_dim, hidden_dim, output_dim)
best_controller.weights = best_weights

starting_parameters = [(1100, 1000, -20), (1300, 800, 0), (1000, 1500, 50), (1100, 1000, -20), (500, 1000, -20),
                       (500, 800, 0), (500, 1500, 50), (500, 400, 25), (500, 400, 50), (500, 800, -50), (500, 800, -25),
                       (500, 800, 0), (500, 800, 25), (500, 800, 50), (500, 1200, -50), (500, 1200, -25), (500, 1200, 0),
                       (500, 1200, 25), (500, 1200, 50), (500, 1600, -50), (500, 1600, -25), (500, 1600, 0),
                       (500, 1600, 25), (500, 1600, 50), (500, 2000, -50), (500, 2000, -25), (500, 2000, 0),
                       (500, 2000, 25), (500, 2000, 50), (800, 400, -50), (800, 400, -25), (800, 400, 0),
                       (800, 400, 25), (800, 400, 50), (800, 800, -50), (800, 800, -25), (800, 800, 0),
                       (800, 800, 25), (800, 800, 50), (800, 1200, -50), (800, 1200, -25), (800, 1200, 0),
                       (800, 1200, 25), (800, 1200, 50), (800, 1600, -50), (800, 1600, -25), (800, 1600, 0),
                       (800, 1600, 25), (800, 1600, 50), (800, 2000, -50), (800, 2000, -25), (800, 2000, 0),
                       (800, 2000, 25), (800, 2000, 50), (1100, 400, -50), (1100, 400, -25), (1100, 400, 0),
                       (1100, 400, 25), (1100, 400, 50), (1100, 800, -50), (1100, 800, -25), (1100, 800, 0),
                       (1100, 800, 25), (1100, 800, 50), (1100, 1200, -50), (1100, 1200, -25), (1100, 1200, 0),
                       (1100, 1200, 25), (1100, 1200, 50), (1100, 1600, -50), (1100, 1600, -25), (1100, 1600, 0),
                       (1100, 1600, 25), (1100, 1600, 50), (1100, 2000, -50), (1100, 2000, -25), (1100, 2000, 0),
                       (1100, 2000, 25), (1100, 2000, 50)]

for fuel, height, velocity in starting_parameters:
    state0 = [fuel, height, velocity]  # Initial state of the lander
    state = state0
    for _ in range(300):  # Run simulation for 100 time steps
        state_norm = [(state[0] - mp0) / mp0, state[1] / h0, state[2] / g]
        inputs = np.array([state_norm])
        F = Fmax * tf.sigmoid(best_controller.forward(inputs)).numpy()[0][0]
        state = lander.simulate(state, int(F))
        # print(f"Still going: Fuel Mass = {state[0]}, Position = {state[1]}, Velocity = {state[2]}")
        if state[1] <= 0 or state[0] <= 0:
            break  # Lander has landed or crashed, stop simulation

    print(f"Landing completed: Fuel Mass = {state[0]}, Position = {state[1]}, Velocity = {state[2]}")
