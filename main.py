import numpy as np
import tensorflow as tf


# Define the lander model and its simulation function
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


# Define the neural network controller
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


# Define the genetic algorithm
class GeneticAlgorithm:
    def __init__(self, population_size, input_dim, hidden_dim, output_dim):
        self.population_size = population_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            controller = NeuralController(self.input_dim, self.hidden_dim, self.output_dim)
            population.append(controller)
        return population

    def select_parents(self, fitness_scores):
        fitness_probs = np.array(fitness_scores) / sum(fitness_scores)
        parents = np.random.choice(range(self.population_size), size=2, replace=False, p=fitness_probs)
        parent1_idx, parent2_idx = parents[0], parents[1]
        return parent1_idx, parent2_idx

    def crossover(self, parent1, parent2):
        child = NeuralController(self.input_dim, self.hidden_dim, self.output_dim)
        for layer in child.weights:
            mask = np.random.choice([0, 1], size=parent1.weights[layer].shape)
            child.weights[layer] = tf.where(mask == 1, parent1.weights[layer], parent2.weights[layer])
        return child

    def mutate(self, child, mutation_rate):
        for layer in child.weights:
            mask = np.random.choice([0, 1], size=child.weights[layer].shape, p=[1 - mutation_rate, mutation_rate])
            noise = np.random.normal(size=child.weights[layer].shape)
            child.weights[layer] = tf.where(mask == 1, child.weights[layer] + noise, child.weights[layer])
        return child

    def evolve_population(self, population, fitness_scores, mutation_rate):
        new_population = []
        for _ in range(self.population_size):
            parent1_idx, parent2_idx = self.select_parents(fitness_scores)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, mutation_rate)
            new_population.append(child)
        return new_population


def calculate_fitness(fitness, state):
    mp, x, v = state
    # if x > 500:
    #     fitness += 2005 - (abs(x) * 2 + abs(v))
    # if x <= 500:
    #     fitness += 1000 - (abs(x) + abs(v))
    # if fitness <= 0 or (x < 100 and abs(v) > 10):
    #     fitness = 1
    # return fitness
    distance_fitness = -abs(x)

    # Velocity fitness
    velocity_fitness = max(0, 10 - abs(v))

    # Combine distance and velocity fitness
    fitness = distance_fitness + velocity_fitness
    return fitness


def main():
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
    hidden_dim = 7
    output_dim = 1
    mutation_rate = 0.3
    generations = 100

    # Initialize the genetic algorithm
    genetic_algorithm = GeneticAlgorithm(population_size, input_dim, hidden_dim, output_dim)
    population = genetic_algorithm.initialize_population()

    for generation in range(generations):
        fitness_scores = []
        i = 0
        for controller in population:
            state = [mp0, h0, v0]  # Initial state of the lander
            fitness = 0
            i += 1
            #print(f"Lander from: Generation = {generation}, Number =  {i}")
            for _ in range(100):  # Run simulation for 100 time steps
                X = np.array(state)
                X_normalized = (X - np.mean(X)) / np.std(X)  # Normalize input
                X_normalized = np.expand_dims(X_normalized, axis=0)  # Add batch dimension
                action = controller.forward(X_normalized).numpy()[0][0]
                #F = Fmax if action > 0.5 else 0
                state_norm = [(state[0] - mp0) / mp0, state[1] / h0, state[2] / g]
                inputs = np.array([state_norm])
                F = Fmax * tf.sigmoid(controller.forward(inputs)).numpy()[0][0]
                state = lander.simulate(state, int(F))
                fitness = calculate_fitness(fitness, state)
                #print(f"Still going: Fuel Mass = {state[0]}, Position = {state[1]}, Velocity = {state[2]}, Fitness = {fitness}")
                if state[1] <= 0 or state[0] <= 0:
                    break  # Lander has landed or crashed, stop simulation
            fitness_scores.append(fitness + 10000)

        best_fitness = max(fitness_scores)
        best_controller = population[fitness_scores.index(best_fitness)]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        if generation != generations - 1:
            population = genetic_algorithm.evolve_population(population, fitness_scores, mutation_rate)

    # Test the best controller
    state = [mp0, h0, v0]  # Initial state of the lander
    for _ in range(300):  # Run simulation for 100 time steps
        X = np.array(state)
        X_normalized = (X - np.mean(X)) / np.std(X)  # Normalize input
        X_normalized = np.expand_dims(X_normalized, axis=0)  # Add batch dimension
        action = best_controller.forward(X_normalized).numpy()[0][0]
        #F = Fmax if action > 0.5 else 0
        state_norm = [(state[0] - mp0) / mp0, state[1] / 1000, state[2] / 40]
        inputs = np.array([state_norm])
        F = Fmax * tf.sigmoid(controller.forward(inputs)).numpy()[0][0]
        state = lander.simulate(state, int(F))
        print(f"Still going: Fuel Mass = {state[0]}, Position = {state[1]}, Velocity = {state[2]}")
        if state[1] <= 0 or state[0] <= 0:
            break  # Lander has landed or crashed, stop simulation

    print(f"Landing completed: Fuel Mass = {state[0]}, Position = {state[1]}, Velocity = {state[2]}")


if __name__ == '__main__':
    main()

