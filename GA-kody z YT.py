import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import random

import tensorflow as tf

class NeuralNetwork:
    def __init__(self, a, b, c, d):
        if isinstance(a, tf.keras.Sequential):
            self.model = a
            self.input_nodes = b
            self.hidden_nodes = c
            self.output_nodes = d
        else:
            self.input_nodes = a
            self.hidden_nodes = b
            self.output_nodes = c
            self.model = self.create_model()

    def copy(self):
        with tf.device('/CPU:0'):
            model_copy = self.create_model()
            weights = self.model.get_weights()
            weight_copies = [tf.identity(w) for w in weights]
            model_copy.set_weights(weight_copies)
            return NeuralNetwork(
                model_copy,
                self.input_nodes,
                self.hidden_nodes,
                self.output_nodes
            )

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_nodes, activation='sigmoid', input_shape=(self.input_nodes,)),
            tf.keras.layers.Dense(self.output_nodes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


class Generation:

    def __init__(self, population):
        self.population = population
        self.species = []
        self.generation = 1
        self.high_score = 0
        self.avg_score = 0
        self.total_score = 0
        self.fitness = 0
        self.progress = 0

    def initialize(self, Creature, width, height):
        for i in range(self.population):
            new_creature = Creature(upper_length=30, upper_width=8, lower_length=30, lower_width=6, x=width * 0.15,
                                    y=height * 0.85, id=i)
            self.species.append(new_creature)

    def pick_one(self):
        index = 0
        r = random.uniform(0, 1)
        while r > 0:
            r -= self.species[index].fitness
            index += 1

        index -= 1

        selected = self.species[index].clone()
        return selected

    def evolve(self, world):
        # Store High Score
        self.generation += 1
        gen_highscore = max([creature.score for creature in self.species])
        self.high_score = max(gen_highscore, self.high_score)

        # Calculate Total Score of this Generation
        total_score = sum([creature.score for creature in self.species])

        # Assign Fitness to each creature
        self.progress = (total_score / self.population) - self.avg_score
        self.avg_score = total_score / self.population
        for i in range(self.population):
            self.species[i].fitness = self.species[i].score / total_score

        # Store new generation temporarily in this array
        new_generation = []

        # Breeding
        for i in range(self.population):
            parentA = self.pick_one()
            parentB = self.pick_one()
            child = parentA.crossover(parentB)
            child.mutate()
            child.id = i
            child.params.id = i
            child.colors = [parentA.colors[0], parentB.colors[1]]
            child.parents = [{'id': parentA.id, 'score': self.species[parentA.id].score},
                             {'id': parentB.id, 'score': self.species[parentB.id].score}]
            new_generation.append(child)

        # Kill Current Generation.
        # i.e. Remove their bodies from MatterJS World and dispose their brain
        for i in range(self.population):
            self.species[i].kill(world)

        # Add new children to the current generation
        self.species = new_generation
        for i in range(self.population):
            self.species[i].add_to_world(world)



