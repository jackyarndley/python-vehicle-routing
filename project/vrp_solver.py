import itertools
import math as math
import operator
import random
import time

import folium
import matplotlib.pyplot as plt
import numpy as np
import openrouteservice as ors
import pandas as pd
from pulp import LpVariable, LpProblem, LpBinary, LpMinimize, lpSum, LpStatus, value

# Load the specific data files into pandas dataframes
data = pd.read_csv('data/new_durations.csv', index_col=0)
data2 = pd.read_csv('data/new_locations.csv', index_col=1)
data3 = pd.read_csv('data/weekdaydemand.csv', index_col=0)

# OpenRouteService key - this is mine
ORS_KEY = '5b3ce3597851110001cf62482926c2987d7f46118f341e666eb30010'

class Location:
    '''
    Location class is to hold information relating to a specific location.
    A route is made up of a sequence of locations.

    Inputs
    --------
    lat : float
        latitude of the location
    long : float
        longitude of the location
    name : string
        name of the location
    demand : integer
        demand of the location
    '''
    def __init__(self, lat, lon, name, demand):
        self.lat = lat
        self.lon = lon
        self.name = name
        self.demand = demand
    
    def distance(self, location):
        '''
        Returns the distance to a another specific location, in this case the distance is
        represented as time.
        '''
        return data[location.name][self.name]

    def nearest_neighbours(self, remaining_locations):
        '''
        Get all of the neighbours of a node and sorts them by distance (time).
        '''
        # Setup empty list to store all of the distances
        distance_results = {}

        # Go through all of the locations and calculate the distance to them
        for i in range(0, len(remaining_locations)):
            distance_results[i] = self.distance(remaining_locations[i])
        
        # Sort them and return the indices
        distances = sorted(distance_results.items(), key = operator.itemgetter(1))
        return [i[0] for i in distances]

class Route:
    '''
    Route class is to hold information relating to a specific route.
    Each route is made up of a sequence of locations.

    Inputs
    --------
    route : list
        list of location objects in route
    distance : float
        total distance (time) of the route
    fitness : string
        fitness of the route (1/distance) in this case
    demand : integer
        total demand across the route
    '''
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
        self.demand = 0
    
    def calc_demand(self):
        '''
        Calculate the total demand across the route.
        '''
        # Go through each location and add the demands together
        route_demand = 0
        for location in self.route:
            route_demand += location.demand
        # Set object property and return
        self.demand = route_demand
        return self.demand

    def calc_distance(self):
        '''
        Calculate the total distance (time) across the route.
        '''
        # Go through each location and add the distances together, also add the distance 
        # between the first and last locations to form complete route.
        route_distance = 0
        for i in range(0, len(self.route)):
            from_location = self.route[i]
            to_location = None
            if i + 1 < len(self.route):
                to_location = self.route[i + 1]
            else:
                to_location = self.route[0]
            route_distance += from_location.distance(to_location)
        # Set object property and return
        self.distance = route_distance
        return self.distance
    
    def calc_fitness(self):
        '''
        Calculate the total fitness of the route. In this case the fitness is 
        calculated as 1/distance, meaning that routes with shorter distance are fitter.
        '''
        # Set object property and return
        self.fitness = 1 / float(self.calc_distance())
        return self.fitness

    def list_path(self):
        '''
        List the entire path of the route in the form of an list of lists with lon and lat
        data in each. This method is primarily for the visualisation of the route using 
        OpenRouteService and the foilum geographic plotting package.
        '''
        # Go through each lon and lat coordinates of each location in the route
        path = [[location.lon, location.lat] for location in self.route]
        # Add the starting location as the end to form the entire route
        return path + [path[0]]

# TODO member functions of this class need commenting
class Solver:
    '''
    Solver class contains all of the information to solve a TSP problem on a selected set 
    of nodes using a genetic algorithm. This algorithm creates successive generations of
    possible routes by sampling, breeding and mutating members from each generation. For 
    a small TSP problem, because there are relatively few different arrangements, this
    algorithm quickly converges to an acceptable solution.

    Inputs
    --------
    locations : list
        list of location objects to be in route
    population_size : integer
        size of each generation to consider
    elite_size : integer
        number of best performing individuals to carry forward to next generation
    mutation_rate : float
        chance of a node swap occuring for each node in the route
    generations : integer
        number of generations of the algorithm to conduct

    Usage
    --------
    To use this, supply the required inputs to the object and use the run method to 
    get the route object representing the TSP solution for the input nodes.
    '''
    def __init__(self, locations, population_size, elite_size, mutation_rate, generations):
        self.locations = locations
        self.population_size = population_size
        self.population = [Route(locations)] * population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def rank_routes(self):
        population_fitness = {i: self.population[i].calc_fitness() for i in range(len(self.population))}
        return sorted(population_fitness.items(), key = operator.itemgetter(1), reverse = True)

    def generate_selection(self, ranked_population):
        df = pd.DataFrame(np.array(ranked_population), columns=["Index","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
        
        selection_results = [ranked_population[i][0] for i in range(0, self.elite_size)]

        for i in range(0, len(ranked_population) - self.elite_size):
            pick = 100*random.random()
            for i in range(0, len(ranked_population)):
                if pick <= df.iat[i,3]:
                    selection_results.append(ranked_population[i][0])
                    break
        return selection_results

    def breed_population(self, mating_pool):
        length = len(mating_pool) - self.elite_size
        pool = random.sample(mating_pool, len(mating_pool))

        children = mating_pool[0:self.elite_size]
        
        for i in range(0, length):
            parent_1 = pool[i]
            parent_2 = pool[len(mating_pool)-i-1]

            selected_gene = sorted([int(random.random() * len(parent_1.route)), int(random.random() * len(parent_1.route))])

            child_part_1 = parent_1.route[selected_gene[0]:selected_gene[1]]
            child_part_2 = [item for item in parent_2.route if item not in child_part_1]

            children.append(Route(child_part_1 + child_part_2))
        self.population = children

    def mutate_population(self):   
        for i in range(len(self.population)):
            individual = self.population[i]
            for swapped in range(len(individual.route)):
                if(random.random() < self.mutation_rate):
                    swapWith = int(random.random() * len(individual.route))
                    
                    location1 = individual.route[swapped]
                    location2 = individual.route[swapWith]
                    
                    individual.route[swapped] = location2
                    individual.route[swapWith] = location1
            self.population[i] = individual

    def next_generation(self):
        population_ranked = self.rank_routes()
        selection_results = self.generate_selection(population_ranked)
        self.breed_population([self.population[i] for i in selection_results])
        self.mutate_population()

    def run(self):
        for _ in range(self.generations):
            self.next_generation()

        return self.population[self.rank_routes()[0][0]]

class Progress:
    '''
    Simple progress bar class to keep track of the algorithm progress

    Inputs
    --------
    max_iterations : integer
        total number of iterations required for full completion of the progress bar
    title : string
        the string to show next to the progress bar
    '''
    def __init__(self, max_iterations, title):
        self.iteration = 0
        self.max_iterations = max_iterations
        self.title = title
        self.start_time = time.time()
        # After the object properties are set, print out the empty progress bar
        print(f'\n{self.title}: [{"-" * 50}] {0:.2f}% ({self.iteration}/{self.max_iterations}) {0:.2f}s   \r', end='')
        
    def increment(self):
        '''
        Increment the progress bar by one iteration and replace current text in console
        with the updated progress.
        '''
        self.iteration += 1
        # Get the fraction complete and then print out update progress
        frac = self.iteration / self.max_iterations
        print(f'{self.title}: [{"#" * int(round(50 * frac)) + "-" * int(round(50 * (1-frac)))}] {100. * frac:.2f}% ({self.iteration}/{self.max_iterations}) {time.time() - self.start_time:.2f}s   \r', end='')

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

    # Current settings are testing to make the algorithm go faster - at the moment there is no mutation
    return Solver(current_locations, 5, 2, 0.0, 5).run()

def generate_routes(demand_locations):
    routes = []
    for current_location in demand_locations:
        remaining_locations = [location for location in demand_locations if location.name not in [current_location.name]]
        distances = current_location.nearest_neighbours(remaining_locations)

        # Get permutations of 5 nearest neighbours and use these to randomise (120)
        permutations = list(itertools.permutations(distances[:5]))
        
        for maximum_capacity in range(current_location.demand, 13):
            for permutation in permutations:
                distances[:5] = permutation
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
                coefficents.append(math.ceil(time * 10) / 10 * 150.0)
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
    return

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
            tooltip = f'{math.ceil(time * 10) / 10:.1f}h ${coefficents[route_index]}',
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

progress = Progress(total_checks * 120, "Generating Routes")

routes = generate_routes(demand_locations)
total_routes = len(routes)

# Adds double the number of routes for the other types of truck
routes.extend(routes)

print("\nSolving Linear Program...")

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