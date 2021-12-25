import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from implementations import all_implementations

data = []
for i in range (300):
    array = np.random.rand(4500)
    temp_array = []
    for sort in all_implementations:
        st = time.time()
        res = sort(array)
        en = time.time()
        temp_array.append(en-st)
    data.append(temp_array)
    
data = pd.DataFrame(data, columns=['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort'])
data.to_csv('data.csv', index=False)
