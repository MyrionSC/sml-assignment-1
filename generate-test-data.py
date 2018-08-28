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
        randomVertex = math.floor(random() * len(buffer))
        edgesFromVertex = buffer[randomVertex].split("\t")
        randomEdge = edgesFromVertex[math.floor(random() * len(edgesFromVertex))]
        realEdges.append((randomVertex, int(randomEdge)))

    ### generate 1000 random edges and test that they are not accidentally real
    for i in range(0, TEST_SIZE_HALF):
        fakeEdges.append(createFakeEdge(buffer))

    ### merge the real and fake edges
    ### todo: not sure if it should be random or not
    edges = realEdges + fakeEdges

    ### save to file
    fileName = "generated-test-data.test"
    with open("data/" + fileName, "w+") as file:
        file.write("Id\tSource\tSink")
        for i, edge in enumerate(edges):
            file.write("\n" + str(i+1) + "\t" + str(edge[0]) + "\t" + str(edge[1]))




def createFakeEdge(buffer):
    fakeEdge = (math.floor(random()*len(buffer)), math.floor(random()*VERTEX_MAX))

    #test if edge is accidentally real
    if fakeEdge[0] < len(buffer):
        vertexEdges = buffer[fakeEdge[0]].split("\t")
        for e in vertexEdges:
            if int(e) == fakeEdge[1]:
                # a real edge was accidentally created, so we try again
                print("real edge hit") # this pretty much never happens
                return createFakeEdge(buffer)

    # edge is not real
    return fakeEdge


if __name__ == '__main__':
    main()

