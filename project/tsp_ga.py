import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

data = pd.read_csv('FoodstuffTravelTimes.csv', index_col=0)
data2 = pd.read_csv('FoodstuffLocations.csv', index_col=1)

class City:
    def __init__(self, lat, lon, name):
        self.lat = lat
        self.lon = lon
        self.name = name
    
    def distance(self, city):
        return data[city.name][self.name]

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def route_distance(self):
        if self.distance ==0:
            path_distance = 0
            for i in range(0, len(self.route)):
                from_city = self.route[i]
                to_city = None
                if i + 1 < len(self.route):
                    to_city = self.route[i + 1]
                else:
                    to_city = self.route[0]
                path_distance += from_city.distance(to_city)
            self.distance = path_distance
        return self.distance
    
    def route_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness

def create_route(city_list):
    return random.sample(city_list, len(city_list))

def initial_population(pop_size, city_list):
    population = []

    for i in range(0, pop_size):
        population.append(create_route(city_list))
    return population

def rank_routes(population):
    fitness_results = {}
    for i in range(0,len(population)):
        fitness_results[i] = Fitness(population[i]).route_fitness()
    return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = True)

def selection(pop_ranked, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(pop_ranked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, elite_size):
        selection_results.append(pop_ranked[i][0])
    for i in range(0, len(pop_ranked) - elite_size):
        pick = 100*random.random()
        for i in range(0, len(pop_ranked)):
            if pick <= df.iat[i,3]:
                selection_results.append(pop_ranked[i][0])
                break
    return selection_results

def mating_pool(population, selection_results):
    matingpool = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    start_gene = min(geneA, geneB)
    end_gene = max(geneA, geneB)

    for i in range(start_gene, end_gene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breed_population(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,elite_size):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if(random.random() < mutation_rate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutate_population(population, mutation_rate):
    mutated_pop = []
    
    for ind in range(0, len(population)):
        mutated_ind = mutate(population[ind], mutation_rate)
        mutated_pop.append(mutated_ind)
    return mutated_pop

def next_generation(current_gen, elite_size, mutation_rate):
    pop_ranked = rank_routes(current_gen)
    selection_Results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_Results)
    children = breed_population(matingpool, elite_size)
    next_generation = mutate_population(children, mutation_rate)
    return next_generation

def genetic_algorithm(population, pop_size, elite_size, mutation_rate, generations):
    pop = initial_population(pop_size, population)
    print("Initial distance: " + str(1 / rank_routes(pop)[0][1]))
    # progress = []
    # progress.append(1 / rank_routes(pop)[0][1])
    
    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)
        # progress.append(1 / rank_routes(pop)[0][1])
        # print(1 / rank_routes(pop)[0][1])
    
    # plt.plot(progress)
    # plt.ylabel('Distance')
    # plt.xlabel('Generation')
    # plt.show()
    # plt.close()

    print("Final distance: " + str(1 / rank_routes(pop)[0][1]))
    bestRouteIndex = rank_routes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def plot_route(route):
    for _, row in data2.iterrows():
        plt.plot(row.Long, row.Lat, 'ko')

    prev_lon = route[0].lon
    prev_lat = route[0].lat

    for i in range(1, len(route)):
        plt.arrow(prev_lon, prev_lat, route[i].lon - prev_lon, route[i].lat - prev_lat, length_includes_head=True, ec='r', fc='r')
        prev_lon = route[i].lon
        prev_lat = route[i].lat

    plt.arrow(prev_lon, prev_lat, route[0].lon - prev_lon, route[0].lat - prev_lat, length_includes_head=True, ec='r', fc='r')

    plt.show()

warehouse_location = City(lat=data2["Lat"]["Warehouse"], lon=data2["Long"]["Warehouse"], name="Warehouse")
demand_nodes = [name for name in data.columns if name not in ["Warehouse"]]

routes = []

for node in demand_nodes:
    target_location = City(lat=data2["Lat"][node], lon=data2["Long"][node], name=node)
    left_nodes = [name for name in demand_nodes if name not in [node]]
    stop_list = [warehouse_location, target_location]
    distance_results = {}
    for i in range(0,len(left_nodes)):
        distance_results[i] = data[left_nodes[i]][node]
    distances = sorted(distance_results.items(), key = operator.itemgetter(1))

    # while route is not up to capacity
    for i, _ in distances[0:5]:
        name = left_nodes[i]
        stop_list.append(City(lat=data2["Lat"][name], lon=data2["Long"][name], name=name))

    routes.append(genetic_algorithm(population=stop_list, pop_size=50, elite_size=5, mutation_rate=0.05, generations=100))

plot_route(routes[0])
plot_route(routes[1])
plot_route(routes[2])
plot_route(routes[3])
plot_route(routes[4])
plot_route(routes[5])
plot_route(routes[6])