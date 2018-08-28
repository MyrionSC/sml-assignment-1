from random import random
import math
TEST_SIZE = 1000
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
    for i in range(0, TEST_SIZE):
        randomVertex = math.floor(random() * len(buffer))
        edgesFromVertex = buffer[randomVertex].split("\t")
        randomEdge = edgesFromVertex[math.floor(random() * len(edgesFromVertex))]
        realEdges.append((randomVertex, int(randomEdge)))

    ### generate 1000 random edges and test that they are not accidentally real
    for i in range(0, 1000):
        fakeEdges.append(createFakeEdge(buffer))

    ### merge the real and fake edges randomly




    ### save to file




def createFakeEdge(buffer):
    fakeEdge = (math.floor(random()*VERTEX_MAX), math.floor(random()*VERTEX_MAX))

    #test if edge is accidentally real
    if fakeEdge[0] < len(buffer):
        vertexEdges = buffer[fakeEdge[0]].split("\t")
        for e in vertexEdges:
            if int(e) == fakeEdge[1]:
                # a real edge was accidentally created, so we try again
                return createFakeEdge(buffer)

    # edge is not real
    return fakeEdge


if __name__ == '__main__':
    main()

