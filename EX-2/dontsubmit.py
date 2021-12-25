import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats as ss

def to_timestamp (d) :
    return d.timestamp()
        

totals = pd.read_csv('dog_rates_tweets.csv').set_index(keys=['id'])
d = totals[totals['text'].str.contains(r'(\d+(\.\d+)?)/10') == True]
d['rating'] = (d['text'].str.extract(r'(\d+(\.\d+)?)/10')[0]).astype(float)
d = d.drop(d[d.rating > 25].index)
d.created_at = pd.to_datetime(d.created_at)
d['timestamp'] = d['created_at'].apply(to_timestamp)

fit = ss.linregress(d['timestamp'], d["rating"].values)

plt.xticks(rotation=25)
plt.plot(d["created_at"].dt.date, d["rating"].values, 'b.', alpha=0.5)
plt.plot(d["created_at"].dt.date, d['timestamp']*fit.slope + fit.intercept, 'r-', linewidth=3)
print(d['timestamp']*fit.slope + fit.intercept)

plt.savefig('temp.png')