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
import seaborn as sns
from pulp import LpVariable, LpProblem, LpBinary, LpMinimize, lpSum, LpStatus, value

# Load the specific data files into pandas dataframes
data = pd.read_csv('data/new_durations.csv', index_col=0)
data2 = pd.read_csv('data/new_locations.csv', index_col=1)
data3 = pd.read_csv('data/weekdaydemand.csv', index_col=0)
data4 = pd.read_csv('data/demandData.csv', index_col=0)

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
        OpenRouteService and the folium geographic plotting package.
        '''
        # Go through each lon and lat coordinates of each location in the route
        path = [[location.lon, location.lat] for location in self.route]
        # Add the starting location as the end to form the entire route
        return path + [path[0]]

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
        '''
        Ranks the possible routes in the population in order based on their fitness. 
        Returns a 2D list with indices and distances.
        '''

        # Get the index and fitness of each member of the population
        population_fitness = {i: self.population[i].calc_fitness() for i in range(len(self.population))}
        # Return the sorted list with the maximum fitness at the beginning
        return sorted(population_fitness.items(), key = operator.itemgetter(1), reverse = True)

    def generate_selection(self, ranked_population):
        '''
        Choose a selection of the population to use to create the next generation.
        Choice is weighted to favour those with high fitness.
        '''
        # Store the fitnesses in a data frame to make a probablistic selection
        fitnesses = pd.DataFrame(np.array(ranked_population), columns=["Index","Fitness"])
        fitnesses['cum_sum'] = fitnesses.Fitness.cumsum()
        fitnesses['cum_perc'] = 100*fitnesses.cum_sum/fitnesses.Fitness.sum()
        
        # Choose the elite members by default
        selection_results = [ranked_population[i][0] for i in range(0, self.elite_size)]

        # Iterate a few more times to get the rest of the selection
        for i in range(0, len(ranked_population) - self.elite_size):
            pick = 100 * random.random()

            # Go through all of the fitnesses array to find the member to select
            for i in range(0, len(ranked_population)):
                if pick <= fitnesses.iat[i,3]:
                    # Add to the selection
                    selection_results.append(ranked_population[i][0])
                    break
        
        return selection_results

    def breed_population(self, mating_pool):
        '''
        Create children based on the current population. Parts of both parents are
        used to create a child with a mixture of each parents routes.
        '''

        # Get the number of children required
        length = len(mating_pool) - self.elite_size

        # Sample randomly from the mating_pool the required number of parents
        pool = random.sample(mating_pool, len(mating_pool))

        # By default select the elite members
        children = mating_pool[0:self.elite_size]
        
        # Iterate over to generate each child
        for i in range(length):
            # Get both parents
            parent_1 = pool[i]
            parent_2 = pool[len(mating_pool)-i-1]

            # Generate the selected gene from the first parent
            selected_gene = sorted([int(random.random() * len(parent_1.route)), int(random.random() * len(parent_1.route))])

            # Get the first and second parts of the child from each parent
            child_part_1 = parent_1.route[selected_gene[0]:selected_gene[1]]
            child_part_2 = [item for item in parent_2.route if item not in child_part_1]

            # Append the child to the children array as a route
            children.append(Route(child_part_1 + child_part_2))
        self.population = children

    def mutate_population(self):
        '''
        Mutates all of the individuals in the population. In the route, there is a
        chance defined by the mutation rate that each node will get randomly swapped with
        another. This is not an essential component of the algorithm, however for large
        TSP problems, this dramatically improves the convergance of the solution.
        '''  

        # Go through each member of the population
        for i in range(len(self.population)):
            individual = self.population[i]

            # Go through each location in the route
            for swapped in range(len(individual.route)):
                # If the individal location is selected to be mutated
                if(random.random() < self.mutation_rate):
                    # Choose the location to swap positions with
                    swapWith = int(random.random() * len(individual.route))
                    
                    # Make the location swap
                    location1 = individual.route[swapped]
                    location2 = individual.route[swapWith]
                    individual.route[swapped] = location2
                    individual.route[swapWith] = location1
            self.population[i] = individual

    def next_generation(self):
        '''
        Runs all of the required methods of the class to generate the next population.
        Population is ranked, selectively breeded and then mutated to get the next
        generation.
        '''

        # Perform the steps required for the new generation to be created
        population_ranked = self.rank_routes()
        selection_results = self.generate_selection(population_ranked)
        self.breed_population([self.population[i] for i in selection_results])
        self.mutate_population()

    def run(self):
        '''
        Runs the next_generation method for the intended number of generations before
        returning the best child (route) from the last generation.
        '''

        # Loop through each generation and update the current population
        for _ in range(self.generations):
            self.next_generation()

        # Return the best population member once all of the generations have been looped through
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
    '''
    Generate a route that is feasable for a defined demand. The locations from the
    current_locations list will be appended to the route until no more routes can 
    possibly be added.

    Inputs
    --------
    maximum_capacity : integer
        maximum capacity of a truck to assume
    current_locations : list
        list of all of the starting locations in the route
    distances : list
        list of all the distances to other locations from target
    remaining_locations : list
        all of the remaining locations that have not been visited by the route
    '''

    # Keep adding nodes to the route until the route is up to capacity
    total_demand = current_locations[1].demand
    j = 0

    # Check that the demand does not exceed capacity and there are locations left
    while total_demand < maximum_capacity and j < len(remaining_locations):
        location = remaining_locations[distances[j]]

        # If the location can be added without exceeding the maximum capacity, add it
        if (total_demand + location.demand) <= maximum_capacity:
            current_locations.append(location)
            total_demand += current_locations[-1].demand
        j += 1

    # Current settings are testing to make the algorithm go fast and perform well
    # At the moment there is no mutation
    return Solver(current_locations, 5, 2, 0.0, 5).run()

def generate_routes(demand_locations):
    '''
    Generate the routes to pass onto the linear optimisation problem. A heuristic algorithm is
    used to reorder the nearest nodes to each location and then the genetic algorithm TSP is used
    to reorder the nodes for a slightly better time.

    Inputs
    --------
    demand_locations : list
        list of all the locations where there is a demand
    '''

    # Create array for routes to be stored
    routes = []
    for current_location in demand_locations:
        remaining_locations = [location for location in demand_locations if location.name not in [current_location.name]]
        distances = current_location.nearest_neighbours(remaining_locations)
        nearest_default = distances[:8]

        # Get permutations of 8 nearest neighbours of length 2 and use these to randomise, 56 in total
        permutations = list(itertools.permutations(distances[:2], 2))
        
        # Vary the maximum capacity of the trucks to generate more diverse solutions
        for maximum_capacity in range(current_location.demand, 13):
            for permutation in permutations:
                # Replace the start of the distances array with the permutation
                distances[:2] = permutation
                distances[2:2] = [index for index in nearest_default if index not in permutation]

                # Run the generate route algorithm with the specified inputs
                routes.append(generate_route(maximum_capacity, [warehouse_location, current_location], distances, remaining_locations))
                progress.increment()

    return routes

def generate_coefficents(routes, total_routes):
    '''
    Generate the coefficents for each route in the linear optimistation problem.
    The algorithm here is based on the conditions that we have been given.

    Inputs
    --------
    routes : list
        list of all the routes that are in the linear program
    total_routes : integer
        the total number of routes that are generated
    '''
    coefficents = []

    # Go through all routes to calculate the coefficents for the objective function
    for route in routes:
        # Calculate the total time in hours
        time = route.calc_distance()
        time += route.calc_demand() * 300
        time /= 3600.0
        
        # Check if the time exceeds four hours
        if time > 4.0:
            if len(coefficents) < total_routes:
                # Cost per 4 hour segment of leased truck
                coefficents.append(1200 * ((time // 4) + 1))
            else:
                # Non-leased schedule should not be allowed if time exceeds 4 hours
                coefficents.append(1000000.0)
        else:
            if len(coefficents) < total_routes:
                # Add the time with ceiling to 6 minute intervals
                coefficents.append(math.ceil(time * 10) / 10 * 150.0)
            else:
                # The truck is a leased truck
                coefficents.append(1200.0)

    return coefficents

def plot_routes_basic(routes, chosen_routes):
    '''
    Generate a basic plot of the data using matplotlib. This displays each location and the 
    direction of the route connecting them. It also displays the demand at each node.

    Inputs
    --------
    routes : list
        list of all the routes that are in the linear program
    chosen_routes : list
        the routes which were chosen for the solution of the linear program
    '''

    # Use the ggplot style and define the correct figures and sizes
    plt.style.use('ggplot')
    _fig, ax1 = plt.subplots(figsize=(20, 15))

    # Go through each location and plot
    for index, row in data2.iterrows():
        ax1.plot(row.Long, row.Lat, 'ko')
        ax1.annotate(f"{data3.demand[index] if index != 'Warehouse' else 0}, {index}", xy=(row.Long, row.Lat), xytext=(row.Long - 0.01, row.Lat + 0.003))

    # Go through each of the chosen routes and plot the path taken
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

    # Save figure with small margins
    plt.savefig("plot1.png", dpi = 300, bbox_inches='tight')

def plot_routes_advanced(routes, chosen_routes):
    '''
    Generate an interactive map of the data using folium, a python wrapper for the 
    leaflet library. The route for each truck is retrieved from OpenRouteService so that a 
    realistic route is shown for each truck path.

    Inputs
    --------
    routes : list
        list of all the routes that are in the linear program
    chosen_routes : list
        the routes which were chosen for the solution of the linear program
    '''

    # Initialise the folium map
    m = folium.Map(location=[warehouse_location.lat, warehouse_location.lon], zoom_start=11)

    # Create a colours lookup dictionary to indicate what colour is assigned to each store type
    colours = {
        "New World": "red",
        "Pak 'n Save": "orange",
        "Four Square": "green",
        "Fresh Collective": "blue"
    }

    # Plot the warehouse location
    folium.Marker([warehouse_location.lat, warehouse_location.lon], popup = warehouse_location.name, icon = folium.Icon(color ='black', prefix='fa', icon='industry')).add_to(m)

    # Plot all of the other demand locations with the correct colours
    for location in demand_locations:
        folium.Marker([location.lat, location.lon], popup = location.name, icon = folium.Icon(color=colours[data2["Type"][location.name]], prefix='fa', icon='shopping-cart')).add_to(m)

    # Create an OpenRouteService client to get route GEOJSON data
    client = ors.Client(key=ORS_KEY)

    # Go through each of the chosen routes
    for route_index in chosen_routes:
        current_route = routes[route_index]

        # Calculate the time of the route in hours
        time = current_route.calc_distance()
        time += current_route.calc_demand() * 300
        time /= 3600.0

        # Randomise the colour of each route using a random hex generator
        color = f'#{"".join(random.choice("0123456789ABCDEF") for i in range(6))}'

        # Get the route which is displayed to the map
        visual_route = client.directions(
            coordinates = current_route.list_path(),
            profile = 'driving-hgv',
            format = 'geojson',
            validate = True
        )

        # Create a polyline object and plot on the map with the correct data and colour
        folium.PolyLine(
            locations = [list(reversed(coord)) for coord in visual_route['features'][0]['geometry']['coordinates']],
            tooltip = f'{math.ceil(time * 10) / 10:.1f}h ${coefficents[route_index]}',
            color = color,
            opacity = 0.75,
            weight = 5
        ).add_to(m)

    # Save the map as an html file
    m.save("routes.html")

# Get the warehouse and other locations stored as location objects
warehouse_location = Location(data2["Lat"]["Warehouse"], data2["Long"]["Warehouse"], "Warehouse", 0)
demand_locations = [Location(data2["Lat"][name], data2["Long"][name], name, data3.demand[name]) for name in data.columns if name not in ["Warehouse"]]

# Calculate the total demand of all of the locations and use this to calculate the total number of routes that will be generated
total_demand = sum([location.demand for location in demand_locations])
total_checks = sum([13 - location.demand for location in demand_locations])

# Initialise the progress bar with the correct number of iterations
progress = Progress(total_checks * 56, "Generating Routes")

# Generate the routes using the generate_routes function
routes = generate_routes(demand_locations)

# Secondary check to get the number of routes
total_routes = len(routes)

# Adds double the number of routes for the leased trucks
routes.extend(routes)

print("\nSolving Linear Program...")

# Load all of the variables into the format required for PuLP
variables = LpVariable.dicts("route", [i for i in range(0, len(routes))], None, None, LpBinary)

# Start PuLP
problem = LpProblem("VRP Optimisation", LpMinimize)

# Generate all of the coefficents for the objective function
coefficents = generate_coefficents(routes, total_routes)

# Input the objective function into PuLP
problem += lpSum([coefficents[i] * variables[i] for i in variables])

# Go through each location to generate the required constraints
for location in demand_locations:
    in_route = np.zeros(len(variables))

    # Use only 0 and 1 variables to indicate if a route passes through a location
    for i in range(0, len(variables)):
        if location in routes[i].route:
            in_route[i] = 1

    # Add the constraint to the linear program
    problem += lpSum([in_route[i] * variables[i] for i in variables]) == 1

# Add in the constraint limiting the total number of non-leased trucks to 20
problem += lpSum([variables[i] for i in range(total_routes)]) <= 20

# Solve the linear optimisation problem using the default solver
problem.solve()

# Print out the status of the solution and the value of the objective
print(f"Status: {LpStatus[problem.status]}")
print(f"Total Cost: ${value(problem.objective)}")

# Due to PuLP's requirements, this code finds the route index from the string that the
# variable is named after, from the variables that are chosen
chosen_routes = [int(route.name.split("_")[1]) for route in problem.variables() if route.varValue > 0.1]
chosen_routes.sort()

print(f"Chosen Routes:")

# Go through each chosen route
for route_index in chosen_routes:
    route_locations = [location.name for location in routes[route_index].route]
    warehouse_index = route_locations.index('Warehouse')
    route_path = route_locations[warehouse_index:] + route_locations[:warehouse_index] + ['Warehouse']
    route_type = 'leased' if route_index >= total_routes else 'default'

    # Print out each route type, cost and path
    print(f"type: {route_type:>7}, cost: {'$' + str(int(coefficents[route_index])):>5}, path: {' -> '.join(route_path)}")

# Plot all of the chosen routes on a matplotlib plot
# plot_routes_basic(routes, chosen_routes)

# Plot all of the chosen routes on an interactive leaflet map
# plot_routes_advanced(routes, chosen_routes)

# Simulation stuff...

# data_copy = data

# for index, row in data.iterrows():
#     # Testing 1.1 times traffic
#     data[index] *= 1.0

costs = []

for i in range(10000):
    # Get new demands
    new_routes = [routes[index] for index in chosen_routes]

    total_cost = 0

    shortages = []

    for route_index in chosen_routes:
        route = routes[route_index]

        for location in [location for location in route.route if location.name not in ["Warehouse"]]:
            location_type = data2["Type"][location.name]

            demands = data4.loc[location_type, :]

            location.demand = random.sample(demands, 1)

        new_demand = route.calc_demand()

        while new_demand > 12:
            least_demand = min([(route.route.index(location), location.demand) for location in route.route if location.name not in ["Warehouse"]], key = operator.itemgetter(1))
            shortages.append(route.route[least_demand[0]])
            route.route.pop(least_demand[0])
            new_demand = route.calc_demand()

        # Calculate the total time in hours
        time = route.calc_distance()
        time += route.calc_demand() * 300
        time /= 3600.0
        
        # Check if the time exceeds four hours
        if time > 4.0:
            if route_index >= total_routes:
                # Cost per 4 hour segment of leased truck
                total_cost += 1200 * ((time // 4) + 1)
            else:
                # Non-leased schedule should not be allowed if time exceeds 4 hours
                total_cost += 600 + math.ceil((time - 4.0) * 10) / 10 * 150.0
        else:
            if route_index < total_routes:
                # Add the time with ceiling to 6 minute intervals
                total_cost += math.ceil(time * 10) / 10 * 150.0
            else:
                # The truck is a leased truck
                total_cost += 1200.0

    for location in shortages:
        total_cost += 1200

    costs.append(total_cost)
    
sns.distplot(costs)
    
plt.show()