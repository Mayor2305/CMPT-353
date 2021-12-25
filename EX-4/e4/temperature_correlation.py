import sys
import numpy as np
import pandas as pd
import difflib as dl
import matplotlib.pyplot as plt
    
def distance(city, stations):    
    R = 6371000
    lat1 = np.radians(city.latitude)
    lat2 = np.radians(stations.latitude)
    delta_lat = np.radians(stations.latitude - city.latitude)
    delta_lon = np.radians(stations.longitude - city.longitude)    
    a = np.sin(delta_lat/2) **2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
    d = R * c
    return d

def best_tmax(city, stations):
    dist = distance(city, stations)
    min_index = dist.values.argmin()
    return stations.avg_tmax[min_index]

stations_file = sys.argv[1]
cityData = sys.argv[2]
save_file = sys.argv[3]

stations = pd.read_json(stations_file, lines=True)
stations.avg_tmax /= 10

cities = pd.read_csv(cityData)

cities.replace("", float("NaN")) #referred to https://www.kite.com/python/answers/how-to-drop-empty-rows-from-a-pandas-dataframe-in-python
cities.dropna(axis = 0, subset = ["population"], inplace = True)
cities.dropna(axis = 0, subset = ["area"], inplace = True)

print(cities)

cities.area /= 1000000
cities['population_density'] = cities.population / cities.area

cities = cities.drop(cities[cities.area > 10000].index)

cities['avg_tmax'] = cities.apply(best_tmax, stations=stations, axis = 1)

print(cities)

plt.scatter(cities.avg_tmax, cities.population_density)
plt.title('Temperature vs Population Density')
plt.xlabel('Avg Max Temperature (\u00b0C)' )
plt.ylabel('Population Density (people/km\u00b2)' )

plt.savefig(save_file)
















