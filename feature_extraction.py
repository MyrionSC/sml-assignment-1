
def jaccard_similarity(src, sink, following_dict, followers_dict):
    src_neighbor_set = set(following_dict[src]) | set(followers_dict[src])
    sink_neighbor_set = set(following_dict[sink]) if sink in following_dict else set() | set(followers_dict[sink])

    intersection = src_neighbor_set & sink_neighbor_set
    union = src_neighbor_set | sink_neighbor_set

    return len(intersection) / len(union)
