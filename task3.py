import random

import matplotlib.pyplot as plt
import numpy as np

# Load the maze from the provided data
maze = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,9,1,1,1,1,1,1,1,1,1,1,9,1,1,1,1,1,1,1,1,1,1,9,1,1,1,1,1,9],
    [9,9,9,9,9,9,1,9,9,9,9,9,1,9,9,9,9,1,9,9,9,9,9,1,9,9,9,9,9,9,9,9,9,9,1,9,1,9,9,1,9,9],
    [1,1,1,1,1,1,1,9,1,1,1,9,1,1,1,9,1,1,9,9,1,9,9,1,1,1,1,1,1,9,1,1,1,1,1,9,1,9,9,1,9,9],
    [1,9,9,9,9,9,9,9,9,9,1,9,9,9,1,1,1,9,9,9,1,9,9,9,9,9,9,9,1,9,1,9,9,9,9,9,1,9,9,1,1,1],
    [1,1,1,1,1,9,1,1,1,1,1,1,1,9,9,9,9,9,9,9,1,9,9,9,9,9,9,9,1,9,1,9,9,9,9,9,1,9,9,9,9,1],
    [9,9,9,9,1,9,9,9,9,1,9,9,9,9,9,9,9,9,1,1,1,1,1,1,1,1,1,1,1,9,1,1,1,1,1,1,1,1,9,9,9,1],
    [1,1,1,9,1,9,9,9,9,1,9,9,9,9,9,9,9,9,1,9,9,9,9,9,9,9,9,9,1,9,9,9,9,9,9,9,9,9,9,9,9,1],
    [1,9,1,9,1,1,1,9,9,1,1,1,1,1,1,1,1,9,1,9,1,1,1,1,1,9,9,9,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,9,1,9,9,9,1,9,9,9,9,9,9,9,9,9,1,9,1,9,1,9,9,9,1,9,9,9,9,9,9,9,9,9,9,9,1,9,9,9,9,9],
    [1,9,1,1,1,1,1,9,1,1,1,9,1,1,1,9,1,9,1,9,1,9,9,1,1,1,1,9,1,1,1,1,9,1,1,1,1,9,1,1,1,1],
    [1,9,9,9,9,9,9,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,9,1,9,9,1,9,1,9,9,1,9,1,9,9,1,9,1,9,9,1],
    [1,9,1,1,1,1,1,9,1,9,1,1,1,9,1,9,1,1,1,9,1,9,9,1,9,9,1,1,1,9,9,1,9,1,1,9,1,9,1,9,9,1],
    [1,9,1,9,9,9,1,9,1,9,9,9,9,9,1,9,9,9,9,9,1,9,9,1,9,9,9,9,9,9,9,1,9,9,9,9,1,1,1,9,9,1],
    [1,9,1,1,1,9,1,1,1,9,1,1,1,1,1,1,1,1,1,1,1,9,9,1,1,1,1,1,1,9,9,1,1,1,1,1,1,9,9,9,1,1]
    
])

# The path is 1 and the obstacles are 9 
START = (0, 0)
END = (13, 40)
POPULATION_SIZE = 100
GENERATIONS = 20
MUTATION_RATE = 0.1

MOVES = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

def create_individual(max_length):
    return [random.randint(0, 3) for _ in range(random.randint(max_length // 2, max_length))]

def create_population(size, max_length):
    return [create_individual(max_length) for _ in range(size)]

def fitness(individual):
    x, y = START
    path_length = 0
    for move in individual:
        dx, dy = MOVES[move]
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < maze.shape[0] and 0 <= new_y < maze.shape[1] and maze[new_x, new_y] != 9:
            x, y = new_x, new_y
            path_length += 1
        if (x, y) == END:
            return 10000 / (path_length + 1)  # Higher reward for reaching the end
    return 1 / (((x - END[0])**2 + (y - END[1])**2) + path_length)

def select_parents(population, fitnesses):
    return random.choices(population, weights=fitnesses, k=2)

def crossover(parent1, parent2):
    split = random.randint(0, min(len(parent1), len(parent2)))
    return parent1[:split] + parent2[split:]

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = random.randint(0, 3)
    return individual

def genetic_algorithm():
    population = create_population(POPULATION_SIZE, maze.shape[0] * maze.shape[1])
    best_fitness = 0
    best_individual = None
    
    for generation in range(GENERATIONS):
        fitnesses = [fitness(ind) for ind in population]
        best_gen_fitness = max(fitnesses)
        best_gen_individual = population[fitnesses.index(best_gen_fitness)]
        
        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            best_individual = best_gen_individual
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
        
        if best_fitness > 9999:  # Solution found
            print(f"Solution found in generation {generation}")
            break
        
        new_population = [best_individual]
        while len(new_population) < POPULATION_SIZE:
            parents = select_parents(population, fitnesses)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    return best_individual

def get_valid_path(individual):
    x, y = START
    path = [START]
    for move in individual:
        dx, dy = MOVES[move]
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < maze.shape[0] and 0 <= new_y < maze.shape[1] and maze[new_x, new_y] != 9:
            x, y = new_x, new_y
            path.append((x, y))
        if (x, y) == END:
            break
    return path

def visualize_path(path):
    maze_copy = maze.copy()
    for x, y in path:
        if (x, y) != START and (x, y) != END:
            maze_copy[x, y] = 2  # Mark the path
    
    plt.figure(figsize=(20, 10))
    plt.imshow(maze_copy, cmap='viridis')
    plt.title("Maze Solution")
    plt.colorbar(ticks=[1, 2, 9], label='Cell Type')
    
    # Highlight the path
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, color='red', linewidth=2, marker='o', markersize=4)
    
    # Mark start and end points
    plt.plot(START[1], START[0], 'go', markersize=12, label='Start')
    plt.plot(END[1], END[0], 'ro', markersize=12, label='End')
    
    plt.legend()
    plt.show()

# Run the genetic algorithm
print("Starting the Genetic Algorithm...")
best_path = genetic_algorithm()
valid_path = get_valid_path(best_path)
print("Path found:", valid_path)
visualize_path(valid_path)