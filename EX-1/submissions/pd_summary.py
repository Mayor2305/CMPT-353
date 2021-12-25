import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

# a.) city with lowest precipitation over the year
sum = totals.sum(axis=1)
minimum_index = sum.argmin()
print("Row with lowest total precipitation:\n", totals.index[minimum_index], sep='')

# b.) Determine the average precipitation in these locations for each month
total_precipitation_month = totals.sum(axis = 0)
total_observations_month = counts.sum(axis = 0)
average_precipitation_month = total_precipitation_month.divide(total_observations_month)
print("Average precipitation in each month:\n", average_precipitation_month, sep='')

# c.) Determine the average precipitation for each city
total_precipitation_city = totals.sum(axis = 1)
total_observations_city = counts.sum(axis = 1)
average_precipitation_city = total_precipitation_city.divide(total_observations_city)
print("Average precipitation in each city:\n", average_precipitation_city, sep='')
