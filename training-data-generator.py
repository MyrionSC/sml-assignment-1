from random import random
import math

TRAINING_FILE_NAME = "generated-training-data.test"
TEST_FILE_NAME = "generated-test-data.test"
GEN_NUM = 20000
GEN_NUM_HALF = int(GEN_NUM / 2)
TEST_NUM = 1000
VERTEX_MAX = 4867136 # This is the number of nodes in train.txt

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
    print("Finding real edges...")
    for i in range(0, GEN_NUM_HALF):
        randomLine = buffer[math.floor(random() * len(buffer))].split("\t")
        source = randomLine[0]
        edge = randomLine[1 + math.floor(random() * len(randomLine) - 1)]
        realEdges.append((int(source), int(edge)))
    print("Real edges found")

    ### generate 1000 random edges and test that they are not accidentally real
    print("Generating fake edges...")
    for i in range(0, GEN_NUM_HALF):
        fakeEdges.append(createFakeEdge(buffer))
    print("Fake edges generated")

    ### merge the real and fake edges
    ### todo: not sure if it should be random or not. For now the first 1000 are real and the second 1000 are fake

    trainingEdges = realEdges[:GEN_NUM_HALF - TEST_NUM] + fakeEdges[:GEN_NUM_HALF - TEST_NUM]
    testEdges = realEdges[GEN_NUM_HALF - TEST_NUM:] + fakeEdges[GEN_NUM_HALF - TEST_NUM:]

    ### save to files
    with open("data/" + TRAINING_FILE_NAME, "w+") as file:
        file.write("Id\tSource\tSink")
        for i, edge in enumerate(trainingEdges):
            file.write("\n" + str(i+1) + "\t" + str(edge[0]) + "\t" + str(edge[1]))
    with open("data/" + TEST_FILE_NAME, "w+") as file:
        file.write("Id\tSource\tSink")
        for i, edge in enumerate(testEdges):
            file.write("\n" + str(i+1) + "\t" + str(edge[0]) + "\t" + str(edge[1]))

    print("-")
    print("Training data written to file: data/" + TRAINING_FILE_NAME)
    print("Test data written to file: data/" + TEST_FILE_NAME)


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

