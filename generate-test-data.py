from random import random
import math
TEST_SIZE = 2000
TEST_SIZE_HALF = int(TEST_SIZE / 2)
VERTEX_MAX = 5000000 # observed from test-public.txt that this is about how large edges get

####         Split in training and test      ####
# 1. Extract 1000 from the real data set
# 2. Create 1000 randomly
# 3. remove the real edges from the random set


def main():

    ### inits
    realEdges = []
    fakeEdges = []
    buffer = ""
    with open("data/train.txt", "r") as train:
        buffer = train.readlines()

    ### extract 1000 real edges from train.txt
    for i in range(0, TEST_SIZE_HALF):
        randomLine = buffer[math.floor(random() * len(buffer))].split("\t")
        source = randomLine[0]
        edge = randomLine[1 + math.floor(random() * len(randomLine) - 1)]
        realEdges.append((int(source), int(edge)))
    print("real edges found")

    ### generate 1000 random edges and test that they are not accidentally real
    for i in range(0, TEST_SIZE_HALF):
        fakeEdges.append(createFakeEdge(buffer))
    print("fake edges generated")

    ### merge the real and fake edges
    ### todo: not sure if it should be random or not. For now the first 1000 are real and the second 1000 are fake
    edges = realEdges + fakeEdges

    ### save to file
    fileName = "generated-test-data.test"
    with open("data/" + fileName, "w+") as file:
        file.write("Id\tSource\tSink")
        for i, edge in enumerate(edges):
            file.write("\n" + str(i+1) + "\t" + str(edge[0]) + "\t" + str(edge[1]))
    print("results written to file: data/" + fileName)


def createFakeEdge(buffer):
    randomLine = buffer[math.floor(random() * len(buffer))].split("\t")
    source = randomLine[0]
    fakeDest = math.floor(random() * VERTEX_MAX)
    #test if edge is accidentally real
    for edge in randomLine[1:]:
        if int(edge) == fakeDest:
            # real edge accidentally hit, try again
            return createFakeEdge(buffer)

    # edge is not real
    return (source, fakeDest)


if __name__ == '__main__':
    main()

