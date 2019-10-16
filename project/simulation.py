import copy
import random
import math
import operator
import pandas as pd

from data import data2, data4
from classes import Location, Solver, Progress

def simulate(routes, chosen_routes, total_routes, total_chosen, samples, traffic_multiplier, progress):
    warehouse_location = Location(data2["Lat"]["Warehouse"], data2["Long"]["Warehouse"], "Warehouse", 0)

    costs = [[] for _ in range(len(traffic_multiplier))]

    for i in range(len(traffic_multiplier)):
        for _j in range(samples):
            total_cost = 0
            shortages = []

            for route_index in chosen_routes:
                route = copy.deepcopy(routes[route_index])

                for location in [location for location in route.route if location.name not in ["Warehouse"]]:
                    location_type = data2["Type"][location.name]

                    demands = data4['Demand'][location_type]

                    location.demand = random.sample(demands, 1)[0]

                route.route = [location for location in route.route if location.name in ["Warehouse"] or location.demand > 0]

                new_demand = route.calc_demand()

                while new_demand > 12:
                    least_demand = min([(route.route.index(location), location.demand) for location in route.route if location.name not in ["Warehouse"]], key = operator.itemgetter(1))
                    shortages.append(copy.deepcopy(route.route[least_demand[0]]))
                    route.route.pop(least_demand[0])
                    new_demand = route.calc_demand()

                # Calculate the total time in hours
                route_time = route.calc_distance(traffic_multiplier[i])
                route_time += route.calc_demand() * 300
                route_time /= 3600.0
                
                # Check if the time exceeds four hours
                if route_time > 4.0:
                    if route_index >= total_routes:
                        # Cost per 4 hour segment of leased truck
                        total_cost += 1200 * ((route_time // 4) + 1)
                    else:
                        # Non-leased schedule should not be allowed if time exceeds 4 hours
                        total_cost += 600 + math.ceil((route_time - 4.0) * 10) / 10 * 200.0
                else:
                    if route_index < total_routes:
                        # Add the time with ceiling to 6 minute intervals
                        total_cost += math.ceil(route_time * 10) / 10 * 150.0
                    else:
                        # The truck is a leased truck
                        total_cost += 1200.0

                # print(f'{total_cost}, {time}, {route.calc_distance()}, {route.calc_demand()}')

            shortage_times = []

            while len(shortages) > 0:
                current_demand = 0
                k = 0
                shortage_indices = []

                while current_demand < 12 and k < len(shortages):
                    shortage_demand = shortages[k].demand

                    if current_demand + shortage_demand <= 12:
                        current_demand += shortage_demand
                        shortage_indices.append(k)
                    k += 1

                shortage_route = Solver([warehouse_location] + [shortages[l] for l in shortage_indices], 5, 2, 0.0, 5).run()

                shortage_time = shortage_route.calc_distance()
                shortage_time += shortage_route.calc_demand() * 300
                shortage_time /= 3600.0

                shortage_times.append(shortage_time)

                for index in sorted(shortage_indices, reverse=True):
                    shortages.pop(index)

            # print(shortage_times)
            for l in range(len(shortage_times)):
                if shortage_times[l] >= 4.0:
                    total_cost += 1200 * ((shortage_times[l] // 4) + 1)
                else:
                    if total_chosen + l <= 20:
                        total_cost += math.ceil(shortage_times[l] * 10) / 10 * 150.0
                    else:
                        total_cost += 1200

            # print(f'{total_cost}')

            costs[i].append(total_cost)
            progress.increment()
    
    return costs