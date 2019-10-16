import pandas as pd

# Load the specific data files into pandas dataframes
data = pd.read_csv('data/new_durations.csv', index_col=0)
data2 = pd.read_csv('data/new_locations.csv', index_col=1)
data3 = pd.read_csv('data/weekdaydemand.csv', index_col=0)
data4 = pd.read_csv('data/daydemand.csv', index_col=0)

for index, row in data4.iterrows():
    row['Demand'] = [int(demand) for demand in row['Demand'].split(' ')]

# This bit is for changing the demands
for index, row in data3.iterrows():
    row['demand'] = row['demand'] - 0

# OpenRouteService key - this is mine
ORS_KEY = '5b3ce3597851110001cf62482926c2987d7f46118f341e666eb30010'