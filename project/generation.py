import itertools
import math

from data import data2
from classes import Location, Solver


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

def generate_routes(demand_locations, progress, search_size):
    '''
    Generate the routes to pass onto the linear optimisation problem. A heuristic algorithm is
    used to reorder the nearest nodes to each location and then the genetic algorithm TSP is used
    to reorder the nodes for a slightly better time.

    Inputs
    --------
    demand_locations : list
        list of all the locations where there is a demand
    progress : object
        progress object for progress bar
    search_size : integer
        search size for extra permuation search (min = 2)
    '''

    warehouse_location = Location(data2["Lat"]["Warehouse"], data2["Long"]["Warehouse"], "Warehouse", 0)

    # Create array for routes to be stored
    routes = []
    for current_location in demand_locations:
        remaining_locations = [location for location in demand_locations if location.name not in [current_location.name]]
        distances = current_location.nearest_neighbours(remaining_locations)
        nearest_default = distances[:8]

        # Get permutations of 8 nearest neighbours of length 2 and use these to randomise, 56 in total
        permutations = list(itertools.permutations(distances[:search_size], 2))
        
        # Vary the maximum capacity of the trucks to generate more diverse solutions
        for maximum_capacity in range(current_location.demand, 13):
            for permutation in permutations:
                # Replace the start of the distances array with the permutation
                distances[:2] = permutation
                distances[2:search_size] = [index for index in nearest_default if index not in permutation]

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
        route_time = route.calc_distance()
        route_time += route.calc_demand() * 300
        route_time /= 3600.0
        
        # Check if the time exceeds four hours
        if route_time > 4.0:
            if len(coefficents) < total_routes:
                # Cost per 4 hour segment of leased truck
                coefficents.append(1200 * ((route_time // 4) + 1))
            else:
                # Non-leased schedule should not be allowed if time exceeds 4 hours
                coefficents.append(1000000.0)
        else:
            if len(coefficents) < total_routes:
                # Add the time with ceiling to 6 minute intervals
                coefficents.append(math.ceil(route_time * 10) / 10 * 150.0)
            else:
                # The truck is a leased truck
                coefficents.append(1200.0)

    return coefficents