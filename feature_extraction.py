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

--- Observation: Each Source for edges in test-public is in train.txt, so it is real.

4. friends in common
    - Hypothesis: For a given edge A -> B, if there exists real edges A -> C and C -> B then
    A -> B is likely real since you are likely to get to know (and thus follow) a friend of a friend at some point
    or have interests in common


'''


import pandas as pd

def extract_followers(followers_dict):
    '''
    edges is a dictionary
    '''
    Followers =pd.Series()
    for key, value in followers_dict.items():
        Followers.loc[key]=len(value)
    return Followers

