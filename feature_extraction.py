'''
Different Features Extractions Methods

1. Number of followers/following for edge A -> B
    in total 4 features: # following A, # following B, # followers A, # followers B --- implemented
2. Number of mutual friends
    different types of "friends" possible like common followers, commmon following --- implemented

3. Distance between friends
    when there are no mutal friends, look at distance between edge,
    so what is the shortest path between the edge
    example for distances:
        direct edge = 1 --- Implemented
        mutal friend = 2
        edge over two friends = 2 e.g. A->B->C-D then A->D 2
        shortest path from Source to Sink
        shortest path from Sink to Source

--- Observation: Each Source for edges in test-public is in train.txt, so it is real.

4. friends in common
    - Hypothesis: For a given edge A -> B, if there exists real edges A -> C and C -> B then
    A -> B is likely real since you are likely to get to know (and thus follow) a friend of a friend at some point
    or have interests in common

5. rank nodes (maybe using markov chain - page rank) and take ranks of sink nodes into account.
    - real users would probably be more likely to follow users with higher rank



'''


import pandas as pd

def reciprocated_follows(src, sink, followers_dict):
    if src in followers_dict and sink in followers_dict and src in followers_dict[sink] and sink in followers_dict[src]:
        # print(str(src) + " and " + str(sink) + " follows each other!")
        return 1
    return 0


## the next 4 functions are all different compinations of following for a--->b with different edge combinations to c
## it always returns the number of people with this specific common connection = len (c)

# a<----c and b<-----c
def same_following(src, sink, following_dict):
    if src in following_dict and sink in following_dict:
        same_following_num = len(set(following_dict[src]) & set(following_dict[sink]))
        # if same_followers_num != 0:
        #     print(str(src) + " and " + str(sink) + " has follows in common: " + str(set(following_dict[src]) & set(following_dict[sink])))
        return same_following_num
    return 0

#a----->c and b------>c
def same_followers(src, sink, followers_dict):
    if src in followers_dict and sink in followers_dict:
        same_followers_num = len(set(followers_dict[src]) & set(followers_dict[sink]))
        # if same_followers_num != 0:
        #     print(str(src) + " and " + str(sink) + " has follows in common: " + str(set(followers_dict[src]) & set(followers_dict[sink])))
        return same_followers_num
    return 0    

#a<-----c and b------>c
def same_following_followers(src, sink,followers_dict, following_dict):
    if src in following_dict and sink in following_dict:
        same_following_num = len(set(following_dict[src]) & set(followers_dict[sink]))
        return same_following_num
    return 0

#a----->c and b<------c
def same_followers_following(src, sink, followers_dict, following_dict):
    if src in following_dict and sink in following_dict:
        same_following_num = len(set(followers_dict[src]) & set(following_dict[sink]))
        return same_following_num
    return 0


def dict_value_len(key, dict):
    if key in dict:
        return len(dict[key])
    return 0