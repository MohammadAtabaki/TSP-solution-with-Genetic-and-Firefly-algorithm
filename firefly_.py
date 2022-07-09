import math
import operator
from numpy.random import random_sample
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
IMG_DIR = 'figures/'

__all__ = ['Ackley', 'Michalewicz']


def generate_population(population_size, problem_dim, min_bound, max_bound):
    error = 1e-10
    data = (max_bound + error - min_bound) * random_sample((population_size, problem_dim)) + min_bound
    data[data > max_bound] = max_bound
    return data


class BaseFunc:
    def __init__(self, dim):
        self.dim = dim
        self.min_bound = np.zeros(self.dim)
        self.max_bound = np.zeros(self.dim)
        self.solution = np.zeros(self.dim)
        self.global_optima = 0
        self.plot_place = 0.25
        self.m = 10
        self.title = ''

    def get_global_optima(self):
        return self.global_optima

    def get_solution(self):
        return self.solution

    def get_search_bounds(self):
        return [self.min_bound, self.max_bound]

    def get_y(self, x):
        return -1

    def plot(self):
        x = np.arange(self.min_bound[0], self.max_bound[0], self.plot_place, dtype=np.float32)
        y = np.arange(self.min_bound[1], self.max_bound[1], self.plot_place, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        Z = []
        for coord in zip(X, Y):
            z = []
            for input in zip(coord[0], coord[1]):
                tmp = list(input)
                tmp.extend(list(self.solution[0:self.dim - 2]))
                z.append(self.get_y(np.array(tmp)))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X, Y, Z)
        plt.show()


class Ackley(BaseFunc):

    def __init__(self, dim):
        super().__init__(dim)
        self.max_bound = np.array([32.768] * self.dim)
        self.min_bound = np.array([-32.768] * self.dim)
        self.solution = np.ones(self.dim)
        self.global_optima = 0
        self.title = 'Ackley'

    def get_y(self, x):
        return 20. - 20. * np.exp(-0.2 * np.sqrt(1. / self.dim * np.sum(np.square(x)))) + np.e - np.exp(
            1. / self.dim * np.sum(np.cos(x * 2. * np.pi)))

    def get_y_2d(self, x, y):
        return 20. - 20. * np.exp(-0.2 * np.sqrt(1. / self.dim * (x ** 2 + y ** 2))) + np.e - np.exp(
            1. / self.dim * (np.cos(x * 2. * np.pi) + np.cos(y * 2. * np.pi)))


class Michalewicz(BaseFunc):

    def __init__(self, dim):
        super().__init__(dim)
        self.max_bound = np.array([np.pi] * self.dim)
        self.min_bound = np.zeros(self.dim)
        self.solution = np.zeros(self.dim)
        self.global_optima = self.get_y(self.solution)
        self.title = 'Michalewicz'
        self.m = 10

    def get_y(self, x):
        y = 0
        for i in range(self.dim):
            y += np.sin(x[i]) * np.power(np.sin((i + 1) * np.power(x[i], 2) / np.pi), 2 * self.m)
        return -y

    def get_y_2d(self, x, y):
        yy = 0
        yy += np.sin(x) * np.power(np.sin((0 + 1) * np.power(x, 2) / np.pi), 2 * self.m)
        yy += np.sin(y) * np.power(np.sin((1 + 1) * np.power(y, 2) / np.pi), 2 * self.m)
        return -yy

# TODO: refactor function firefly_optimisation function call
class Firefly:

    def __init__(self, problem_dim, min_bound, max_bound):
        self.func = Michalewicz(problem_dim)
        self.position = generate_population(1, problem_dim, min_bound, max_bound)[0]
        self.brightness = None
        self.update_brightness()

    # the best fit is 0
    def update_brightness(self):
        self.brightness = -self.func.get_y(self.position)


class FireflyOptimizer:

    def __init__(self, **kwargs):
        self.population_size = int(kwargs.get('population_size', 10))
        self.problem_dim = kwargs.get('problem_dim', 2)
        self.min_bound = kwargs.get('min_bound', -5)
        self.max_bound = kwargs.get('max_bound', 5)
        self.generations = kwargs.get('generations', 10)
        self.population = self._population(self.population_size, self.problem_dim, self.min_bound, self.max_bound)
        self.gamma = kwargs.get('gamma', 0.97)  # absorption coefficient
        self.alpha = kwargs.get('alpha', 0.25)  # randomness [0,1]
        self.beta_init = kwargs.get('beta_init', 1)
        self.beta_min = kwargs.get('beta_min', 0.2)
        self.optimization_benchmark = kwargs.get('optimization_benchmark', 'Ackley')

    @staticmethod
    def _population(population_size, problem_dim, min_bound, max_bound):
        population = []
        for i in range(population_size):
            population.append(Firefly(problem_dim, min_bound, max_bound))
        return population

    def step(self):
        self.population.sort(key=operator.attrgetter('brightness'), reverse=True)
        self._modify_alpha()
        tmp_population = self.population
        for i in range(self.population_size):
            for j in range(self.population_size):
                if self.population[i].brightness > tmp_population[j].brightness:
                    r = math.sqrt(np.sum((self.population[i].position - tmp_population[j].position) ** 2))
                    beta = (self.beta_init - self.beta_min) * math.exp(-self.gamma * r ** 2) + self.beta_min
                    tmp = self.alpha * (np.random.random_sample((1, self.problem_dim))[0] - 0.5) * (
                            self.max_bound - self.min_bound)
                    self.population[j].position = self.check_position(
                        self.population[i].position * (1 - beta) + tmp_population[
                            j].position * beta + tmp)
                    self.population[j].update_brightness()
        self.population[0].position = generate_population(1, self.problem_dim, self.min_bound, self.max_bound)[0]
        self.population[0].update_brightness()

    def run_firefly(self):
        for t in range(self.generations):
            print('Generation %s, best fitness %s' % (t, self.population[0].brightness))
            self.step()
        self.population.sort(key=operator.attrgetter('brightness'), reverse=True)
        return self.population[0].brightness, self.population[0].position

    def check_position(self, position):
        position[position > self.max_bound] = self.max_bound
        position[position < self.min_bound] = self.min_bound
        return position

    def _modify_alpha(self):
        delta = 1 - (10 ** (-4) / 0.9) ** (1 / self.generations)
        self.alpha = (1 - delta) * self.alpha
