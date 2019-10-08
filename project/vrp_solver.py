import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt, itertools
from pulp import *
import folium, openrouteservice as ors

data = pd.read_csv('data/FoodstuffTravelTimes.csv', index_col=0)
data2 = pd.read_csv('data/FoodstuffLocations.csv', index_col=1)
data3 = pd.read_csv('data/weekdaydemand.csv', index_col=0)

# dont look at this
ORS_KEY = '5b3ce3597851110001cf62482926c2987d7f46118f341e666eb30010'

class Location:
    def __init__(self, lat, lon, name, demand):
        self.lat = lat
        self.lon = lon
        self.name = name
        self.demand = demand
    
    def distance(self, city):
        return data[city.name][self.name]

    def nearest_neighbours(self, remaining_locations):
        distance_results = {}

        for i in range(0, len(remaining_locations)):
            distance_results[i] = data[remaining_locations[i].name][self.name]
        
        distances = sorted(distance_results.items(), key = operator.itemgetter(1))
        return [i[0] for i in distances]

class Route:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
        self.demand = 0
    
    def calc_demand(self):
        route_demand = 0
        for location in self.route:
            route_demand += location.demand
        self.demand = route_demand
        return self.demand

    def calc_distance(self):
        if self.distance == 0:
            route_distance = 0
            for i in range(0, len(self.route)):
                from_location = self.route[i]
                to_location = None
                if i + 1 < len(self.route):
                    to_location = self.route[i + 1]
                else:
                    to_location = self.route[0]
                route_distance += from_location.distance(to_location)
            self.distance = route_distance
        return self.distance
    
    def calc_fitness(self):
        self.fitness = 1 / float(self.calc_distance())
        return self.fitness

    def list_path(self):
        path = [[location.lon, location.lat] for location in self.route]
        return path + [path[0]]

class Progress:
    def __init__(self, max_iterations, title):
        self.iteration = 0
        self.max_iterations = max_iterations
        self.title = title
        print(f'\n{self.title}: [{"-" * 50}] {0:.2f}% ({self.iteration}/{self.max_iterations})\r', end='')
        
    def increment(self):
        self.iteration += 1
        frac = self.iteration / self.max_iterations
        print(f'{self.title}: [{"#" * int(round(50 * frac)) + "-" * int(round(50 * (1-frac)))}] {100. * frac:.2f}% ({self.iteration}/{self.max_iterations})\r', end='')

def generate_population(size, locations):
    population = []

    for _i in range(0, size):
        population.append(Route(locations))
    return population

def rank_routes(population):
    fitness_results = {}
    for i in range(0,len(population)):
        fitness_results[i] = population[i].calc_fitness()
    return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = True)

def generate_selection(ranked_population, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(ranked_population), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, elite_size):
        selection_results.append(ranked_population[i][0])
    for i in range(0, len(ranked_population) - elite_size):
        pick = 100*random.random()
        for i in range(0, len(ranked_population)):
            if pick <= df.iat[i,3]:
                selection_results.append(ranked_population[i][0])
                break
    return selection_results

def generate_mating_pool(population, selection_results):
    mating_pool = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        mating_pool.append(population[index])
    return mating_pool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1.route))
    geneB = int(random.random() * len(parent1.route))
    
    start_gene = min(geneA, geneB)
    end_gene = max(geneA, geneB)

    for i in range(start_gene, end_gene):
        childP1.append(parent1.route[i])
        
    childP2 = [item for item in parent2.route if item not in childP1]

    child = childP1 + childP2
    return Route(child)

def breed_population(mating_pool, elite_size):
    children = []
    length = len(mating_pool) - elite_size
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(0,elite_size):
        children.append(mating_pool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(mating_pool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutation_rate):
    for swapped in range(len(individual.route)):
        if(random.random() < mutation_rate):
            swapWith = int(random.random() * len(individual.route))
            
            location1 = individual.route[swapped]
            location2 = individual.route[swapWith]
            
            individual.route[swapped] = location2
            individual.route[swapWith] = location1
    return individual

def mutate_population(population, mutation_rate):
    mutated_population = []
    
    for i in range(0, len(population)):
        mutated_individual = mutate(population[i], mutation_rate)
        mutated_population.append(mutated_individual)
    return mutated_population

def next_generation(current_generation, elite_size, mutation_rate):
    ranked_population = rank_routes(current_generation)
    selection_results = generate_selection(ranked_population, elite_size)
    mating_pool = generate_mating_pool(current_generation, selection_results)
    children = breed_population(mating_pool, elite_size)
    next_generation = mutate_population(children, mutation_rate)
    return next_generation

def genetic_algorithm(locations, population_size, elite_size, mutation_rate, generations):
    population = generate_population(population_size, locations)
    
    for _i in range(0, generations):
        population = next_generation(population, elite_size, mutation_rate)

    return population[rank_routes(population)[0][0]]

def generate_route(maximum_capacity, current_locations, distances, remaining_locations):
    # while route is not up to capacity
    total_demand = current_locations[1].demand
    j = 0
    while total_demand < maximum_capacity and j < len(remaining_locations):
        location = remaining_locations[distances[j]]

        if (total_demand + location.demand) <= maximum_capacity:
            current_locations.append(location)
            total_demand += current_locations[-1].demand
        j += 1

    return genetic_algorithm(current_locations, 20, 5, 0.05, 25)

def generate_routes(demand_locations):
    routes = []
    for current_location in demand_locations:
        remaining_locations = [location for location in demand_locations if location.name not in [current_location.name]]
        distances = current_location.nearest_neighbours(remaining_locations)

        # Get permutations of 4 nearest neighbours and use these to randomise (24)
        permutations = list(itertools.permutations(distances[:4]))
        
        for maximum_capacity in range(current_location.demand, 13):
            for permutation in permutations:
                distances[:4] = permutation
                routes.append(generate_route(maximum_capacity, [warehouse_location, current_location], distances, remaining_locations))
                progress.increment()

    return routes

def generate_coefficents(routes, total_routes):
    coefficents = []

    for route in routes:
        time = route.calc_distance()
        time += route.calc_demand() * 300
        time /= 3600.0
        
        if time > 4.0:
            if len(coefficents) < total_routes:
                # Cost per 4 hour segment
                coefficents.append(1200 * ((time // 4) + 1))
            else:
                # Route should not be allowed
                coefficents.append(1000000.0)
        else:
            if len(coefficents) < total_routes:
                coefficents.append(round(time, 1) * 150.0)
            else:
                coefficents.append(1200.0)

    return coefficents

def plot_routes_basic(routes, chosen_routes):
    for index, row in data2.iterrows():
        ax1.plot(row.Long, row.Lat, 'ko')
        ax1.annotate(f"{data3.demand[index] if index != 'Warehouse' else 0}, {index}", xy=(row.Long, row.Lat), xytext=(row.Long - 0.01, row.Lat + 0.003))

    for i in chosen_routes:
        current_route = routes[i]
        color = (random.random(), random.random(), random.random())
        prev_lon = current_route.route[0].lon
        prev_lat = current_route.route[0].lat

        for i in range(1, len(current_route.route)):
            next_lon = current_route.route[i].lon
            next_lat = current_route.route[i].lat
            plt.arrow(prev_lon, prev_lat, next_lon - prev_lon, next_lat - prev_lat, length_includes_head=True, ec=color, fc=color)
            prev_lon = next_lon
            prev_lat = next_lat

        ax1.arrow(prev_lon, prev_lat, current_route.route[0].lon - prev_lon, current_route.route[0].lat - prev_lat, length_includes_head=True, ec=color, fc=color)

    plt.savefig("plot1.png", dpi = 300, bbox_inches='tight')
    plt.show()

def plot_routes_advanced(routes, chosen_routes):
    m = folium.Map(location=[warehouse_location.lat, warehouse_location.lon], zoom_start=11)

    colours = {
        "New World": "red",
        "Pak 'n Save": "orange",
        "Four Square": "green",
        "Fresh Collective": "blue"
    }

    folium.Marker([warehouse_location.lat, warehouse_location.lon], popup = warehouse_location.name, icon = folium.Icon(color ='black', prefix='fa', icon='industry')).add_to(m)

    for location in demand_locations:
        folium.Marker([location.lat, location.lon], popup = location.name, icon = folium.Icon(color=colours[data2["Type"][location.name]], prefix='fa', icon='shopping-cart')).add_to(m)

    client = ors.Client(key=ORS_KEY)

    for route_index in chosen_routes:
        current_route = routes[route_index]

        time = current_route.calc_distance()
        time += current_route.calc_demand() * 300
        time /= 3600.0

        color = f'#{"".join(random.choice("0123456789ABCDEF") for i in range(6))}'

        visual_route = client.directions(
            coordinates = current_route.list_path(),
            profile = 'driving-hgv',
            format = 'geojson',
            validate = True
        )

        folium.PolyLine(
            locations = [list(reversed(coord)) for coord in visual_route['features'][0]['geometry']['coordinates']],
            tooltip = f'{time:.1f}h ${coefficents[route_index]}',
            color = color,
            opacity = 0.75,
            weight = 5
        ).add_to(m)

    m.save("routes.html")
    return

warehouse_location = Location(data2["Lat"]["Warehouse"], data2["Long"]["Warehouse"], "Warehouse", 0)
demand_locations = [Location(data2["Lat"][name], data2["Long"][name], name, data3.demand[name]) for name in data.columns if name not in ["Warehouse"]]

total_demand = sum([location.demand for location in demand_locations])
total_checks = sum([13 - location.demand for location in demand_locations])

progress = Progress(total_checks * 24, "Generating Routes")

routes = generate_routes(demand_locations)
total_routes = len(routes)

# Adds double the number of routes for the other types of truck
routes.extend(routes)

# [f'{i}_{"default" if i < total_routes else "extra"}'

variables = LpVariable.dicts("route", [i for i in range(0, len(routes))], None, None, LpBinary)

problem = LpProblem("VRP Optimisation", LpMinimize)

coefficents = generate_coefficents(routes, total_routes)

problem += lpSum([coefficents[i] * variables[i] for i in variables])

for location in demand_locations:
    in_route = np.zeros(len(variables))

    for i in range(0, len(variables)):
        if location in routes[i].route:
            in_route[i] = 1

    problem += lpSum([in_route[i] * variables[i] for i in variables]) == 1

problem += lpSum([variables[i] for i in range(total_routes)]) <= 20

print("\nSolving Linear Program...")
problem.writeLP("vrp_optimisation.lp")
problem.solve()

print(f"Status: {LpStatus[problem.status]}")
print(f"Total Cost: ${value(problem.objective)}")

plt.style.use('ggplot')
fig, ax1 = plt.subplots(figsize=(20, 15))

chosen_routes = [int(route.name.split("_")[1]) for route in problem.variables() if route.varValue > 0.1]
chosen_routes.sort()

print(f"Chosen Routes:")
for route_index in chosen_routes:
    route_locations = [location.name for location in routes[route_index].route]
    warehouse_index = route_locations.index('Warehouse')
    route_path = route_locations[warehouse_index:] + route_locations[:warehouse_index] + ['Warehouse']
    route_type = 'leased' if route_index >= total_routes else 'default'

    print(f"type: {route_type:>7}, cost: {'$' + str(int(coefficents[route_index])):>5}, path: {' -> '.join(route_path)}")

plot_routes_basic(routes, chosen_routes)

plot_routes_advanced(routes, chosen_routes)