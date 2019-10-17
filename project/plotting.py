import random
import math
import folium
import matplotlib.pyplot as plt
import openrouteservice as ors

from data import data, data2, data3, data4, ORS_KEY
from classes import Location

def plot_routes_basic(routes, chosen_routes, file_name):
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
    plt.savefig(file_name, dpi = 300, bbox_inches='tight')
    plt.close()

def plot_routes_advanced(routes, chosen_routes, coefficents, file_name):
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

    # Get the warehouse and other locations stored as location objects
    warehouse_location = Location(data2["Lat"]["Warehouse"], data2["Long"]["Warehouse"], "Warehouse", 0)
    demand_locations = [Location(data2["Lat"][name], data2["Long"][name], name, data3.demand[name]) for name in data.columns if name not in ["Warehouse"]]

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
        route_time = current_route.calc_distance()
        route_time += current_route.calc_demand() * 300
        route_time /= 3600.0

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
            tooltip = f'{math.ceil(route_time * 10) / 10:.1f}h ${coefficents[route_index]}',
            color = color,
            opacity = 0.75,
            weight = 5
        ).add_to(m)

    # Save the map as an html file
    m.save(file_name)