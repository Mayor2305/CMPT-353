import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

# a.) city with lowest precipitation over the year
sum = np.sum(totals, axis = 1)
minimum_index = np.argmin(sum)
print("Row with lowest total precipitation:\n", minimum_index, sep='')

# b.) Determine the average precipitation in these locations for each month
total_precipitation_month = np.sum(totals, axis = 0)
total_observations_month = np.sum(counts, axis = 0)
average_precipitation_month = np.divide(total_precipitation_month, total_observations_month)
print("Average precipitation in each month:\n", average_precipitation_month, sep='')

# c.) Determine the average precipitation for each city
total_precipitation_city = np.sum(totals, axis = 1)
total_observations_city = np.sum(counts, axis = 1)
average_precipitation_city = np.divide(total_precipitation_city, total_observations_city)
print("Average precipitation in each city:\n", average_precipitation_city, sep='')

# d.) Calculate the total precipitation for each quarter in each city
reshaped_array = np.reshape(totals, (4 * (totals.shape[0]), 3))
sum_reshape_array = np.sum(reshaped_array, axis = 1, keepdims=1)
re_reshape_array = np.reshape(sum_reshape_array, ((totals.shape[0]), 4))
print("Quarterly precipitation totals:\n", re_reshape_array, sep='')
