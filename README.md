Travelling Salesman Problem Optimization using Genetic and Firefly Algorithms

Project Overview:

This project aims to solve the Traveling Salesman Problem (TSP) using a combination of Genetic Algorithms (GA) and Firefly Algorithm (FA). The goal is to find the shortest possible route that visits all given cities exactly once and returns to the starting point.

Date of Completion:

9th July 2022

Project Structure:

The project consists of multiple files and directories:

1. firefly_.py

This file implements the Firefly Algorithm for optimization. It includes:

. Benchmark functions (Ackley, Michalewicz) for optimization evaluation.

. Firefly class that represents individual fireflies.

. FireflyOptimizer class that executes the FA with parameters like population size, generations, absorption coefficient (gamma), and randomness factor (alpha).

2. main.py

This is the main script to run the TSP optimization using GA. It includes:

. Path class to represent and evaluate TSP paths.

. GA pipeline that evolves a population of TSP solutions over multiple generations using selection, crossover, and mutation.

. Hyperparameter tuning using Hyperopt to optimize GA parameters.

. Functions to read and process TSP datasets, generate distance matrices, and visualize the results.

3. GA_tsp_optimisation/ (Folder)

This folder contains the core GA components:

. __init__.py: Initializes the module.

. crossover.py: Implements different crossover techniques (PMX, Ordered, Cycle).

. mutation.py: Implements mutation strategies (Reverse Sequence Mutation - RSM).

. selection.py: Implements selection methods (Roulette Wheel selection).

Results and Visualization:

. The optimized TSP route is displayed using Matplotlib.

. GA Convergence Graph shows the improvement in path length over generations.
