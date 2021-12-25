import sys
import numpy as np
import pandas as pd
import difflib as dl

def ratings(data):
    return rating_file_read.rating[data]
    
def find_rating(data):
    # data.apply
    d = dl.get_close_matches(data, rating_file_read.title, n=len(rating_file_read.title), cutoff=0.6)
    index = rating_file_read[rating_file_read.title.isin(d)].index
    count = len(index)
    rating = 0
    rating = list(map(ratings, index.values))
    average = 0
    if count != 0:
        average = sum(rating) / count
       
    return average
    

movie_list = sys.argv[1]
rating_file = sys.argv[2]
save_file = sys.argv[3]

movie_file_read = open(movie_list)
movie_data = movie_file_read.readlines()
movie_data = list(map(str.strip, movie_data)) # referred to https://stackoverflow.com/questions/7984169/remove-trailing-newline-from-the-elements-of-a-string-list/7984202
movie_df = pd.DataFrame(movie_data, columns=['title'])


rating_file_read = pd.read_csv(rating_file)

movie_df['rating'] = movie_df['title'].apply(find_rating).round(2)

movie_df = movie_df.sort_values('title')
movie_df = movie_df.drop(movie_df[movie_df.rating == 0].index)

movie_df.to_csv(save_file, index=False)



movie_file_read.close()