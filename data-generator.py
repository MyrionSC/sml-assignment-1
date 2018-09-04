import pandas as pd
import random
import math
from random import normalvariate
from scipy.stats import norm

import helper


FILE_NAME = "generated-data.txt"
GEN_NUM = 200000
GEN_NUM_HALF = int(GEN_NUM / 2)
VERTEX_MAX = 4867136 # This is the number of nodes in train.txt


def gen_edges(following_dict, following_num_list, sorted_following_num_list, mean, std):
    realEdges, fakeEdges = [], []

    ### extract GEN_NUM_HALF real edges from train.txt
    print("Finding real edges...")
    progress_percent, progress_mod = 0, math.floor(GEN_NUM_HALF / 10)
    for i in range(0, GEN_NUM_HALF):
        while True:
            # source = normal_choice(sorted_following_num_list, mean, std)[0]
            source = random.choice(list(following_dict.keys()))
            if len(following_dict[source]) != 0:
                break

        edge = random.choice(following_dict[source])
        realEdges.append((source, edge, True))

        if i % progress_mod == 0:
            print("Progress: " + str(progress_percent) + "%")
            progress_percent += 10
    # print("Real edges found")

    ### generate GEN_NUM_HALF random edges and test that they are not accidentally real
    print("Generating fake edges...")

    # read test-public.txt into dict and use it to make sure that non of the fake edges are accidentally real
    test_public_df = pd.read_csv("data/test-public.txt", sep="\t")
    test_public_dict = dict(zip(list(test_public_df.Source), list(test_public_df.Sink)))

    progress_percent = 0
    for i in range(0, GEN_NUM_HALF):
        fakeEdges.append(createFakeEdge(following_dict, sorted_following_num_list, mean, std, test_public_dict))
        if i % progress_mod == 0:
            print("Progress: " + str(progress_percent) + "%")
            progress_percent += 10
    print("Fake edges generated")

    ### merge the real and fake edges
    return realEdges + fakeEdges


def dict_value_len_list(keys, dict):
    values = []
    for k in keys:
        if k in dict:
            values.append(len(dict[k]))
    return values

def main():
    print("Generating " + str(GEN_NUM) + " edges (half real, half fake)")

    print("Loading data files...")
    following_dict = helper.read_file_to_dict("./data/train.txt")

    following_num_list = []
    for key, value in following_dict.items():
        following_num_list.append((key, len(value)))

    sorted_following_num_list = sorted(following_num_list, key=lambda x: x[1])

    # mean, std = norm.fit([x[1] for x in sorted_following_num_list])
    # print("mean and standard deviation of train.txt: " + str((mean, std)))
    mean, std = 15890, 8000

    print("Load done")
    print()

    edges = gen_edges(following_dict, following_num_list, sorted_following_num_list, mean, std)

    # i = 0
    # while True:
    #     edges = gen_edges(following_dict, following_num_list, sorted_following_num_list, mean, std)
    #     print()
    #     sources_follows_num = [len(following_dict[x[0]]) for x in edges]
    #     s_m,s_d = norm.fit(sources_follows_num)
    #     sink_follows_num = dict_value_len_list([x[1] for x in edges], following_dict)
    #     i_m,i_d = norm.fit(sink_follows_num)
    #     print("gen-data sources following_num mean, std: " + str((s_m,s_d)))
    #     print("gen-data sink following_num mean, std: " + str((i_m,i_d)))
    #     print()
    #     print("test-public sources following_num mean, std: 1329.491, 10111.3652867414")
    #     print("test-public sink following_num mean, std: 1821.3625, 27640.438630728957")
    #
    #     print(i)
    #     i += 1
    #     if math.fabs(s_m - 1329) < 100 and math.fabs(s_d - 10111) < 500:
    #         break


    ### save to files
    with open("data/" + FILE_NAME, "w+") as file:
        file.write("Id\tSource\tSink\tLabel")
        for i, edge in enumerate(edges):
            file.write("\n" + str(i+1).strip() + "\t" + str(edge[0]).strip() + "\t" + str(edge[1]).strip() +
                       "\t" + str(edge[2]).strip())

    print("-")
    print("Generated data written to file: data/" + FILE_NAME)


def createFakeEdge(following_dict, sorted_following_num_list, mean, std, test_public_dict):
    while True:
        # source = normal_choice(sorted_following_num_list, mean, std)[0]
        source = random.choice(list(following_dict.keys()))
        if len(following_dict[source]) != 0:
            break

    fakeDest = random.randrange(0, VERTEX_MAX + 1)
    #test if edge is accidentally real
    for edge in following_dict[source]:
        if int(edge) == fakeDest:
            # real edge accidentally hit, try again
            return createFakeEdge(following_dict, sorted_following_num_list, mean, std, test_public_dict)
        elif source in test_public_dict and test_public_dict[source] == int(edge):
            # accidentally hit edge in test_public which might be real
            return createFakeEdge(following_dict, sorted_following_num_list, mean, std, test_public_dict)

    # edge is not real
    return source, fakeDest, False

def normal_choice(lst, mean, stddev):
    while True:
        index = int(normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return lst[index]

if __name__ == '__main__':
    main()

