import os
from datetime import datetime

def read_file (filename):
    edges = dict() 
    with open(filename) as trainfile:
        for i, line in enumerate(trainfile):
            line_list = [int(k) for k in line.split("\t")]
            edges[line_list[0]] = line_list[1:]
    return edges



    ### Save predictions to file
    ### prediction is a list of values
    # write_file(predictions)

def write_file (predictions):
    filename = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".csv"
    os.makedirs("output", exist_ok=True)

    with open("output/" + filename, "w") as file:
        file.write("Id,Prediction")
        for i in range(0, len(predictions)):
            # save random predictions as baseline. Should be about 50% correct
            file.write("\n" + str(i + 1) + "," + predictions[i])
            print("output written to file: output/" + filename)
