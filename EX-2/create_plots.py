import sys
import pandas as pd
import matplotlib.pyplot as plt

filename1 = sys.argv[1]
filename2 = sys.argv[2]

fileRead1 = pd.read_csv(filename1, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])

fileRead2 = pd.read_csv(filename2, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])

fileRead1 = fileRead1.sort_values(by = 'views', ascending=False)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(fileRead1['views'].values, 'b-') 
plt.title("Popularity Distribution")
plt.xlabel("Rank")
plt.ylabel("Views")
plt.subplot(1, 2, 2)
mergedFile = pd.merge(fileRead1, fileRead2, on = 'page') # refered to https://www.youtube.com/watch?v=XMjSGGej9y8
plt.plot(mergedFile['views_x'].values, mergedFile['views_y'].values, 'bo')
plt.xscale("log")
plt.yscale("log")
plt.title("Hourly Correlation")
plt.xlabel("Hour 1 views")
plt.ylabel("Hour 2 views")
plt.savefig('wikipedia.png')