import numpy as np
import tensorflow as tf
import os
import pickle
import gen_data

os.makedirs('./saved_controller', exist_ok=True)


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
        weights = {'W1': tf.Variable(tf.random.normal([self.input_dim, self.hidden_dim])),
                   'W2': tf.Variable(tf.random.normal([self.hidden_dim, self.output_dim]))}
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

    def tournament_selection(self, fitness_scores, tournament_size):
        selected_parents = []
        population_size = len(fitness_scores)
        for _ in range(population_size):
            tournament_indices = np.random.choice(range(population_size), size=tournament_size, replace=False)
            tournament_fitness_scores = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness_scores)]
            selected_parents.append(winner_index)
        return selected_parents

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
        selected_parents = self.tournament_selection(fitness_scores, tournament_size=16)
        for _ in range(self.population_size):
            parent1_idx, parent2_idx = np.random.choice(selected_parents, size=2, replace=False)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, mutation_rate)
            new_population.append(child)
        return new_population


def calculate_fitness(fitness, state):
    mp, x, v = state
    wx = 1
    wv = 2
    wmp = 1
    fitness = -wx * x + -wv * abs(v)
    if x == 0 and abs(v) < 5:
        fitness += 1000
    if x == 0 and abs(v) > 30:
        fitness = -20000
    return fitness


def generate_tuples(fuel_min, fuel_max, fuel_step, position_min, position_max, position_step, velocity_min,
                    velocity_max, velocity_step, list_of_tuples):
    for k in range(5):
        for i in range(fuel_min, fuel_max + 1, fuel_step):
            for j in range(position_min, position_max + 1, position_step):
                for z in range(velocity_min, velocity_max + 1, velocity_step):
                    list_of_tuples.append((i, j, z))

    return list_of_tuples


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
    hidden_dim = 14
    output_dim = 1
    mutation_rate = 0.01
    generations = 30
    generation = 0

    fuel_min = 200
    fuel_max = 1300
    fuel_step = 400
    position_min = 200
    position_max = 2000
    position_step = 400
    velocity_min = -50
    velocity_max = 50
    velocity_step = 10
    list_of_tuples = []

    # Initialize the genetic algorithm
    genetic_algorithm = GeneticAlgorithm(population_size, input_dim, hidden_dim, output_dim)
    population = genetic_algorithm.initialize_population()
    starting_parameters = generate_tuples(fuel_min, fuel_max, fuel_step, position_min, position_max, position_step,
                                          velocity_min,
                                          velocity_max, velocity_step, list_of_tuples)

    for fuel, position, velocity in starting_parameters:
        # for generation in range(generations):
        fitness_scores = []
        i = 0
        for controller in population:
            state0 = [fuel, position, velocity]
            state = state0  # Initial state of the lander
            fitness = 0
            i += 1
            # print(f"Lander from: Generation = {generation}, Number =  {i}")
            for _ in range(100):  # Run simulation for 100 time steps
                state_norm = [(state[0] - state0[0]) / state0[0], state[1] / state0[1], state[2] / g]
                inputs = np.array([state_norm])
                F = Fmax * tf.sigmoid(controller.forward(inputs)).numpy()[0][0]
                state = lander.simulate(state, int(F))
                fitness = calculate_fitness(fitness, state)
                # print(
                #     f"Still going: Generation = {generation}, Number =  {i}, Fuel Mass = {state[0]}, Position = {state[1]}, Velocity = {state[2]}, Fitness = {fitness}")
                # if _ == 99:
                    # print(
                    #     f"Still going: Generation = {generation}, Number =  {i}, Fuel Mass = {state[0]}, Position = {state[1]}, Velocity = {state[2]}, Fitness = {fitness}")
                if state[1] <= 0 or state[0] <= 0:
                    # print(
                    #     f"Still going: Generation = {generation}, Number =  {i}, Fuel Mass = {state[0]}, Position = {state[1]}, Velocity = {state[2]}, Fitness = {fitness}")
                    break  # Lander has landed or crashed, stop simulation
            fitness_scores.append(fitness)

        best_fitness = max(fitness_scores)
        best_controller = population[fitness_scores.index(best_fitness)]
        best_weights = best_controller.weights
        file_path = f'./saved_controller/best_controller_generation_{generation}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(best_weights, f)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
        generation += 1
        if generation != len(starting_parameters) - 1:
        # if generation != generations - 1:
            population = genetic_algorithm.evolve_population(population, fitness_scores, mutation_rate)

    # Test the best controller
    state = [900, 600, 10]  # Initial state of the lander
    for _ in range(300):  # Run simulation for 100 time steps
        state_norm = [(state[0] - mp0) / mp0, state[1] / h0, state[2] / g]
        inputs = np.array([state_norm])
        F = Fmax * tf.sigmoid(best_controller.forward(inputs)).numpy()[0][0]
        state = lander.simulate(state, int(F))
        print(f"Still going: Fuel Mass = {state[0]}, Position = {state[1]}, Velocity = {state[2]}")
        if state[1] <= 0 or state[0] <= 0:
            break  # Lander has landed or crashed, stop simulation

    print(f"Landing completed: Fuel Mass = {state[0]}, Position = {state[1]}, Velocity = {state[2]}")


if __name__ == '__main__':
    main()
