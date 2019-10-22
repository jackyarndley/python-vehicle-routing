import math
import copy
import random
import operator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pulp import LpVariable, LpProblem, LpBinary, LpMinimize, lpSum, LpStatus, value

from data import data, data2, data3, data4, data5, data6
from classes import Location, Route, Solver, Progress
from generation import generate_routes, generate_coefficents
from simulation import simulate_weekday, simulate_weekend
from plotting import plot_routes_basic, plot_routes_advanced

if __name__ == '__main__':
    print("Running for Weekdays...")
    
    # Get the warehouse and other locations stored as location objects
    warehouse_location = Location(data2["Lat"]["Warehouse"], data2["Long"]["Warehouse"], "Warehouse", 0)
    demand_locations = [Location(data2["Lat"][name], data2["Long"][name], name, data3.demand[name]) for name in data.columns if name not in ["Warehouse"]]

    # Calculate the total demand of all of the locations and use this to calculate the total number of routes that will be generated
    total_demand = sum([location.demand for location in demand_locations])
    total_checks = sum([13 - location.demand for location in demand_locations])

    search_size = 8

    # Initialise the progress bar with the correct number of iterations
    progress = Progress(total_checks * int((math.factorial(search_size) / math.factorial(search_size - 2))), "Generating Routes")

    # Generate the routes using the generate_routes function
    routes = generate_routes(demand_locations, progress, search_size)

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
    total_chosen = len(chosen_routes)

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
    plot_routes_basic(routes, chosen_routes, "plot1.png")

    # Plot all of the chosen routes on an interactive leaflet map
    plot_routes_advanced(routes, chosen_routes, coefficents, "routes1.html")

    samples = 2500
    traffic_multiplier = [1.4]

    progress = Progress(samples * len(traffic_multiplier), "Simulating Weekdays without Traffic")
    
    # Calculating costs for weekday simulations
    costs = simulate_weekday(routes, chosen_routes, total_routes, total_chosen, samples, traffic_multiplier, progress)
    
    print("\nPlotting Simulation...")

    # Plotting and saving simulation distributions
    plt.style.use('ggplot')
    _fig, ax1 = plt.subplots(figsize=(10, 7))

    i = 0
    for cost in costs:
        labelstr = 'Traffic Multiplier: ' + str(traffic_multiplier[i])
        sns.distplot(cost, bins = 50, ax = ax1, label=labelstr)
        i = i + 1

    ax1.set_xlabel('Cost ($)')
    ax1.set_ylabel('Density')
    ax1.axvline(np.percentile(costs,2.5),color='r')
    ax1.axvline(np.percentile(costs,97.5),color='r',label='95% Prediction Range')
    ax1.legend()
    plt.savefig("plot3.png", dpi = 300, bbox_inches='tight')
    plt.close()

    traffic_multiplier = [1,1.2,1.4,1.6]
    progress = Progress(samples * len(traffic_multiplier), "Simulating Weekdays with Traffic")

    # Calculating costs for weekday simulations
    costs = simulate_weekday(routes, chosen_routes, total_routes, total_chosen, samples, traffic_multiplier, progress)

    print("\nPlotting Simulation...")

    # Plotting and saving simulation distributions
    plt.style.use('ggplot')
    _fig, ax1 = plt.subplots(figsize=(10, 7))

    i = 0
    for cost in costs:
        labelstr = 'Traffic Multiplier: ' + str(traffic_multiplier[i])
        sns.distplot(cost, bins = 50, ax = ax1, label=labelstr)
        i = i + 1
    
    ax1.set_xlabel('Cost ($)')
    ax1.set_ylabel('Density')
    ax1.legend()
    plt.savefig("plot4.png", dpi = 300, bbox_inches='tight')
    plt.close()

    for i in range(len(traffic_multiplier)):
        print(f'Multiplier: {traffic_multiplier[i]:.2f}, 2.5-97.5 Cost Percentile: {np.percentile(costs[i], [2.5, 97.5])}')

    print("\nRunning for Saturdays...")

    # Get the warehouse and other locations stored as location objects
    warehouse_location = Location(data2["Lat"]["Warehouse"], data2["Long"]["Warehouse"], "Warehouse", 0)
    demand_locations = [Location(data2["Lat"][name], data2["Long"][name], name, data5.demand[name]) for name in data.columns if name not in ["Warehouse"]]

    # Calculate the total demand of all of the locations and use this to calculate the total number of routes that will be generated
    total_demand = sum([location.demand for location in demand_locations])
    total_checks = sum([13 - location.demand for location in demand_locations])

    search_size = 8

    # Initialise the progress bar with the correct number of iterations
    progress = Progress(total_checks * int((math.factorial(search_size) / math.factorial(search_size - 2))), "Generating Routes")

    # Generate the routes using the generate_routes function
    routes = generate_routes(demand_locations, progress, search_size)

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
    total_chosen = len(chosen_routes)

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
    plot_routes_basic(routes, chosen_routes, "plot2.png")

    # Plot all of the chosen routes on an interactive leaflet map
    plot_routes_advanced(routes, chosen_routes, coefficents, "routes2.html")

    traffic_multiplier = [1.2]
    progress = Progress(samples * len(traffic_multiplier), "Simulating Saturdays without Traffic")

    # Calculating costs for Saturday simulation
    costs = simulate_weekend(routes, chosen_routes, total_routes, total_chosen, samples, traffic_multiplier, progress)

    print("\nPlotting Simulation...")

    # Plotting and saving simulation distributions
    plt.style.use('ggplot')
    _fig, ax1 = plt.subplots(figsize=(10, 7))

    i = 0
    for cost in costs:
        labelstr = 'Traffic Multiplier: ' + str(traffic_multiplier[i])
        sns.distplot(cost, bins = 50, ax = ax1, label=labelstr)
        i = i + 1

    ax1.set_xlabel('Cost ($)')
    ax1.set_ylabel('Density')
    ax1.axvline(np.percentile(costs,2.5),color='r')
    ax1.axvline(np.percentile(costs,97.5),color='r',label='95% Prediction Range')
    ax1.legend()
    plt.savefig("plot5.png", dpi = 300, bbox_inches='tight')
    plt.close()

    traffic_multiplier = [1,1.2,1.4]
    progress = Progress(samples * len(traffic_multiplier), "Simulating Saturdays with Traffic")

    # Calculating costs for Saturday simulation with traffic
    costs = simulate_weekend(routes, chosen_routes, total_routes, total_chosen, samples, traffic_multiplier, progress)

    print("\nPlotting Simulation...")

    # Plotting and saving simulation distributions
    plt.style.use('ggplot')
    _fig, ax1 = plt.subplots(figsize=(10, 7))

    i = 0
    for cost in costs:
        labelstr = 'Traffic Multiplier: ' + str(traffic_multiplier[i])
        sns.distplot(cost, bins = 50, ax = ax1, label=labelstr)
        i = i + 1
    
    ax1.set_xlabel('Cost ($)')
    ax1.set_ylabel('Density')
    ax1.legend()
    plt.savefig("plot6.png", dpi = 300, bbox_inches='tight')
    plt.close()

    for i in range(len(traffic_multiplier)):
        print(f'Multiplier: {traffic_multiplier[i]:.2f}, 2.5-97.5 Cost Percentile: {np.percentile(costs[i], [2.5, 97.5])}')