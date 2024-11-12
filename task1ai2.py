import operator
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# City class definition
class City:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
    
    def distance(self, other):
        xDis = abs(self.x - other.x)
        yDis = abs(self.y - other.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return self.name

# Fitness class definition
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
    
    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(len(self.route)):
                fromCity = self.route[i]
                toCity = self.route[(i + 1) % len(self.route)]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

# Helper functions
def createRoute(cityList):
    return random.sample(cityList, len(cityList))

def initialPopulation(popSize, cityList):
    return [createRoute(cityList) for _ in range(popSize)]

def rankRoutes(population):
    fitnessResults = {i: Fitness(route).routeFitness() for i, route in enumerate(population)}
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
    
    eliteSize = min(eliteSize, len(popRanked))
    selectionResults = [popRanked[i][0] for i in range(eliteSize)]
    for _ in range(len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    return [population[i] for i in selectionResults]

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    childP1 = parent1[startGene:endGene]
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = matingpool[:eliteSize]
    pool = random.sample(matingpool, len(matingpool))
    for i in range(len(matingpool) - eliteSize):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))
            individual[swapped], individual[swapWith] = individual[swapWith], individual[swapped]
    return individual

def mutatePopulation(population, mutationRate):
    return [mutate(ind, mutationRate) for ind in population]

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for _ in range(generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

# Create the list of cities with random coordinates
city_names = ['Berlin', 'London', 'Moscow', 'Barcelona', 'Rome', 'Paris', 'Vienna', 'Munich', 'Istanbul', 'Kyiv', 
              'Bucharest', 'Minsk', 'Warsaw', 'Budapest', 'Milan', 'Prague', 'Sofia', 'Birmingham', 'Brussels', 'Amsterdam']

cityList = [City(name, random.randint(0, 200), random.randint(0, 200)) for name in city_names]

# Test different combinations of population sizes and mutation rates
population_sizes = [10, 20, 50, 100]
mutation_rates = [0.9, 0.6, 0.3, 0.1]

results = {}

for pop_size in population_sizes:
    for mut_rate in mutation_rates:
        print(f"Testing with population size: {pop_size}, mutation rate: {mut_rate}")
        
        best_route = geneticAlgorithm(population=cityList, popSize=pop_size, eliteSize=min(20, pop_size), mutationRate=mut_rate, generations=500)
        best_distance = Fitness(best_route).routeDistance()
        
        results[(pop_size, mut_rate)] = best_distance
        
        print(f"Best route: {' -> '.join(city.name for city in best_route)}")
        print(f"Best distance: {best_distance}")
        print("--------------------")

# Print all results
for (pop_size, mut_rate), distance in results.items():
    print(f"Population size: {pop_size}, Mutation rate: {mut_rate}, Best distance: {distance}")

# Visualize the results
fig, ax = plt.subplots(figsize=(12, 8))

for pop_size in population_sizes:
    distances = [results[(pop_size, mut_rate)] for mut_rate in mutation_rates]
    ax.plot(mutation_rates, distances, marker='o', label=f'Population Size: {pop_size}')

ax.set_xlabel('Mutation Rate')
ax.set_ylabel('Best Distance')
ax.set_title('Effect of Population Size and Mutation Rate on TSP Solution')
ax.legend()
plt.xscale('log')
plt.grid(True)
plt.show()