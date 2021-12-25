import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
from statsmodels.stats.multicomp import pairwise_tukeyhsd


data = pd.read_csv('data.csv')
anova = stats.f_oneway(data.qs1, data.qs2, data.qs3, data.qs4, data.qs5, data.merge1, data.partition_sort)
print(anova.pvalue)
x_melt = pd.melt(data)
posthoc = pairwise_tukeyhsd(
    x_melt['value'], x_melt['variable'],
    alpha=0.05)
print(posthoc)
fig = posthoc.plot_simultaneous()
print('\n\naccording to posthoc.plot_simultaneous() \nspeed: partition_sort > qs1 > qs5 > qs4 > qs2 > qs3/merge1.\nindestingushable are: qs3 and merge1')