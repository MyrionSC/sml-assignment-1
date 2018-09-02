from random import random
import math

FILE_NAME = "generated-data.txt"
GEN_NUM = 2000000
GEN_NUM_HALF = int(GEN_NUM / 2)
VERTEX_MAX = 4867136 # This is the number of nodes in train.txt

def main():

    ### inits
    realEdges = []
    fakeEdges = []
    buffer = ""
    with open("data/train.txt", "r") as train:
        buffer = train.readlines()

    ### extract GEN_NUM_HALF real edges from train.txt
    print("Finding real edges...")
    for i in range(0, GEN_NUM_HALF):
        randomLine = buffer[math.floor(random() * len(buffer))].split("\t")
        source = randomLine[0]
        edge = randomLine[1 + math.floor(random() * len(randomLine) - 1)]
        realEdges.append((int(source), int(edge), True))
    print("Real edges found")

    ### generate GEN_NUM_HALF random edges and test that they are not accidentally real
    print("Generating fake edges...")
    for i in range(0, GEN_NUM_HALF):
        fakeEdges.append(createFakeEdge(buffer))
    print("Fake edges generated")

    ### merge the real and fake edges
    edges = realEdges + fakeEdges

    ### save to files
    with open("data/" + FILE_NAME, "w+") as file:
        file.write("Id\tSource\tSink\tLabel")
        for i, edge in enumerate(edges):
            file.write("\n" + str(i+1).strip() + "\t" + str(edge[0]).strip() + "\t" + str(edge[1]).strip() +
                       "\t" + str(edge[2]).strip())

    print("-")
    print("Generated data written to file: data/" + FILE_NAME)


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
    return source, fakeDest, False


if __name__ == '__main__':
    main()

