from random import random
import math
TEST_SIZE = 1000
VERTEX_MAX = 5000000

####         Split in training and test      ####
# 1. Extract 1000 from the real data set
# 2. Create 1000 randomly
# 3. remove the real edges from the random set

### inits
realEdges = []
fakeEdges = []
buffer = ""
with open("data/train.txt", "r") as train:
    buffer = train.readlines()


### extract 1000 edges from train.txt
# for i in range(0, TEST_SIZE):
for i in range(0, 10):
    randomVertex = math.floor(random() * len(buffer))
    edgesFromVertex = buffer[randomVertex].split("\t")
    randomEdge = edgesFromVertex[math.floor(random() * len(edgesFromVertex))]
    realEdges.append((randomVertex, int(randomEdge)))

print(realEdges)


### generate 1000 random edges
for i in range(0, 10):
    fakeEdges.append((math.floor(random()*len(buffer)), math.floor(random()*len(buffer))))

print(fakeEdges)


### test that these edges are not in train.txt (proper fake edges)



### save to file



