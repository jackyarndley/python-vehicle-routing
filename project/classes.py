import pandas as pd
import numpy as np
import random
import time
import operator

from data import data, data2, data3, data4

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

    def calc_distance(self, multiplier = 1.0):
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
        return self.distance * multiplier
    
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