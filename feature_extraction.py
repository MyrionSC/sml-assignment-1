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

def reciprocated_follows(src, sink, followers_dict):
    if src in followers_dict and sink in followers_dict and src in followers_dict[sink] and sink in followers_dict[src]:
        # print(str(src) + " and " + str(sink) + " follows each other!")
        return 1
    return 0

def same_following(src, sink, following_dict):
    if src in following_dict and sink in following_dict:
        same_followers_num = len(set(following_dict[src]) & set(following_dict[sink]))
        # if l != 0:
        #     print(str(src) + " and " + str(sink) + " has follows in common: " + str(set(following_dict[src]) & set(following_dict[sink])))
        return same_followers_num
    return 0







## this does not really make sense
# if two users follow each other, they both get a 1, otherwise 0
# def extract_mutual_follows(followers_dict, following_dict):
#     ### does it make sense that this is a series?
#     mutual_follows = pd.Series()
#
#     # print(len(followers_dict))
#     # i = 0
#     for user, followers in followers_dict.items():
#         for f in followers:
#             if user in following_dict[f]:
#                 mutual_follows.loc[user] = f
#                 mutual_follows.loc[f] = user
#                 # print("mutual followers found: " + str(len(mutual_follows)))
#         # print(i)
#         # i += 1
#
#     return mutual_follows


