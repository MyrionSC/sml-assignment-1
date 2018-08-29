'''
Different Features Extractions Methods

1. Number of followers/following for edge A -> B
    in total 4 features: # following A, # following B, # followers A, # followers B
2. Number of mutual friends
    different types of "friends" possible like common followers, commmon following

3. Distance between friends
    when there are no mutal friends, look at distance between edge,
    so what is the shortest path between the edge
    example for distances:
        direct edge = 1
        mutal friend = 2
        edge over two friends = 2 e.g. A->B->C-D then A->D 2
'''
import pandas as pd

def get_followers(edges):
    '''
    edges is a dictionary
    '''
    numbers =pd.Series()
    for key, value in edges.items():
        numbers.loc[key]=len(value)
    return numbers

