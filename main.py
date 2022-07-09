from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
from scipy.spatial.distance import euclidean
from GA_tsp_optimisation import Selector, Crossover, Mutation
import operator

coordinates = None
matrix = None


class Path:
    def __init__(self, path):
        self.path = path
        self.fitness = _evaluate_fitness(path)
        self._prob = None

    def update_path(self, new_path):
        self.path = new_path
        self.fitness = _evaluate_fitness(new_path)


def _evaluate_fitness(path):
    dist = 0
    for i in range(len(path) - 1):
        if i == (len(path) - 1):
            dist += matrix[path[0]][path[i + 1]]
            break
        dist += matrix[path[i + 1]][path[i]]
    return dist


def _generate_population(num_of_cities, population_size):
    population = []
    for _ in range(population_size):
        path = np.random.permutation([i for i in range(num_of_cities)])
        population.append(Path(path))
        # draw_path(path, coordinates)
    return population


def ga_pipeline(mat=None, population_size=20, generations=200, best_perc=0.2,
                mutation_probability=0.2, mutation_intensity=0.3,
                verbose=1, coord=None, plot=0):
    num_of_cities = mat.shape[0]
    global matrix
    matrix = mat
    global coordinates
    coordinates = coord
    population = _generate_population(num_of_cities, population_size)

    s = Selector(selection_type='roulette')
    c = Crossover(crossover_type='ordered')
    m = Mutation(mutation_type='rsm')

    x, y = [], []

    for ii in range(generations):
        population.sort(key=operator.attrgetter('fitness'), reverse=False)
        print('')

        new_generation = []
        for i in range(int(population_size * best_perc)):
            new_generation.append(population[i])
        pairs_generator = s.selection(population=population, best_perc=best_perc)
        for i, j in pairs_generator:
            child_1, child_2 = c.crossover(parent_1=i.path, parent_2=j.path)
            new_generation.append(Path(child_1))
            new_generation.append(Path(child_2))
        population = new_generation[:population_size]
        for i in range(1, len(population)):
            population[i].update_path(m.mutation(population[i].path, mutation_probability=mutation_probability))
        population.sort(key=operator.attrgetter('fitness'), reverse=False)
        if verbose:
            print('========== generation %s ==========' % ii)
            print('best so far: %s\n' % population[0].fitness)
        x.append(ii)
        y.append(population[0].fitness)
        if plot:
            if ii % 500 == 0:
                draw_path(population[0].path, coordinates, ii)
    draw_convergence(x, y, 'ps = %s, bp = %s, mr = %s, mi = %s' % (
        round(population_size, 2), round(best_perc, 2), round(mutation_probability, 2), round(mutation_intensity, 2)))
    return population[0].fitness


def draw_path(path, coords, iteration):
    for i in range(len(path) - 1):
        x = [coords[path[i]][0], coords[path[i + 1]][0]]
        y = [coords[path[i]][1], coords[path[i + 1]][1]]
        plt.plot(x, y, marker='o', markersize=2)
    plt.plot([coords[path[len(path) - 1]][0], coords[path[0]][0]], [coords[path[len(path) - 1]][1], coords[path[0]][1]],
             marker='o', markersize=2)
    plt.title('Mininmal path on %s iteration' % iteration)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def draw_convergence(x_list, y_list, params):
    plt.plot(x_list, y_list)
    plt.xlabel('Iteration')
    plt.ylabel('Minimal distance')
    plt.title('GA convergence %s' % params)
    plt.show()


def read_tsp_file(fname):
    with open('./data/%s' % fname) as f:
        while f.readline() != 'TYPE : TSP\n':
            pass
        points_count = int(f.readline().split(':')[1])
        while f.readline() != 'NODE_COORD_SECTION\n':
            pass
        coord = np.zeros(shape=(points_count, 2))
        for i in range(points_count):
            coord[i] = f.readline().split()[1:]
    return coord


def create_matrix(fname):
    coord = read_tsp_file(fname)
    full_name = './data/%s_matrix.npy' % fname.split()[0]
    try:
        matrix = np.load(full_name)
    except IOError:
        matrix = np.zeros((len(coord), len(coord)))
        for i in range(len(coord)):
            for j in range(len(coord)):
                matrix[i][j] = euclidean(coord[i], coord[j])
        np.save(full_name, matrix)
    return matrix


fname = argv[1]

matrix = create_matrix(fname)
coord = read_tsp_file(fname)


# 564

def ga_wrapper(params):
    population_size = int(params['population_size'])
    best_perc = params['best_perc']
    mutation_rate = params['mutation_rate']
    val = ga_pipeline(mat=matrix, population_size=population_size,
                      best_perc=best_perc, mutation_probability=mutation_rate,
                      generations=400, verbose=0, plot=0)
    print(params, val)
    return {'loss': val, 'status': STATUS_OK}


def hyperopt_optimization():
    trials = Trials()

    space = {
        'best_perc': hp.uniform('best_perc', 0.1, 0.9),
        'population_size': hp.quniform('population_size', 20, 50, 1),
        'mutation_rate': hp.uniform('mutation_rate', 0.1, 0.8)
    }

    best = fmin(
        ga_wrapper,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials
    )

    return best


def run_tsp(population_size, best_perc, mutation_probability, generations, coord):
    val = ga_pipeline(mat=matrix, population_size=population_size,
                      best_perc=best_perc, mutation_probability=mutation_probability,
                      generations=generations, verbose=1, coord=coord, plot=1)
    return val


def main():
    print(hyperopt_optimization())
    print(
        run_tsp(mutation_probability=0.4, best_perc=0.3,
                population_size=40, generations=1000, coord=coord))


if __name__ == '__main__':
    main()
