import pandas as pd
from random import random
import math

FILE_NAME = "generated-data.txt"
GEN_NUM = 20000
GEN_NUM_HALF = int(GEN_NUM / 2)
VERTEX_MAX = 4867136 # This is the number of nodes in train.txt

def main():
    print("Generating " + str(GEN_NUM) + " edges (half real, half fake)")

    ### inits
    realEdges = []
    fakeEdges = []
    buffer = ""
    with open("data/train.txt", "r") as train:
        buffer = train.readlines()

    ### extract GEN_NUM_HALF real edges from train.txt
    print("Finding real edges...")
    progress_percent, progress_mod = 0, math.floor(GEN_NUM_HALF / 10)
    for i in range(0, GEN_NUM_HALF):
        randomLine = buffer[math.floor(random() * len(buffer))].split("\t")
        source = randomLine[0]
        edge = randomLine[1 + math.floor(random() * len(randomLine) - 1)]
        realEdges.append((int(source), int(edge), True))
        if i % progress_mod == 0:
            print("Progress: " + str(progress_percent) + "%")
            progress_percent += 10
    print("Real edges found")

    ### generate GEN_NUM_HALF random edges and test that they are not accidentally real
    print("Generating fake edges...")

    # read test-public.txt into dict and use it to make sure that non of the fake edges are accidentally real
    test_public_df = pd.read_csv("data/test-public.txt", sep="\t")
    test_public_dict = dict(zip(list(test_public_df.Source), list(test_public_df.Sink)))

    progress_percent = 0
    for i in range(0, GEN_NUM_HALF):
        fakeEdges.append(createFakeEdge(buffer, test_public_dict))
        if i % progress_mod == 0:
            print("Progress: " + str(progress_percent) + "%")
            progress_percent += 10
    print("Fake edges generated")

    ### merge the real and fake edges
    edges = realEdges + fakeEdges

    ### save to files
    with open("data/" + FILE_NAME, "w+") as file:
        file.write("Id\tSource\tSink\tLabel")
        for i, edge in enumerate(edges):
            file.write("\n" + str(i+1).strip() + "\t" + str(edge[0]).strip() + "\t" + str(edge[1]).strip() +
                       "\t" + str(edge[2]).strip())

    # print("-")
    # print("Generated data written to file: data/" + FILE_NAME)


def createFakeEdge(buffer, test_public_dict):
    randomLine = buffer[math.floor(random() * len(buffer))].split("\t")
    source = int(randomLine[0])
    fakeDest = math.floor(random() * VERTEX_MAX)
    #test if edge is accidentally real
    for edge in randomLine[1:]:
        if int(edge) == fakeDest:
            # real edge accidentally hit, try again
            return createFakeEdge(buffer, test_public_dict)
        elif source in test_public_dict and test_public_dict[source] == int(edge):
            # accidentally hit edge in test_public which might be real
            return createFakeEdge(buffer, test_public_dict)

    # edge is not real
    return source, fakeDest, False


if __name__ == '__main__':
    main()

