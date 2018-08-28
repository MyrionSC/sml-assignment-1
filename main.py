#!/usr/bin/python3

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from random import random
from datetime import datetime
import models

def main ():
    ### Inits
    vertex_set = set()
    sink_dict = {}

    ### load training data
    # with open("./data/train.txt") as trainfile:
    #     for i, line in enumerate(trainfile):
    #         line_list = [int(k) for k in line[:-1].split("\t")]
    #         vertex_set.add(line_list[0])
    #         for s in line_list[1:]:
    #             if s in sink_dict:
    #                 sink_dict[s] += 1
    #             else:
    #                 sink_dict[s] = 1
    #         if i % 1000 == 0:
    #             print(i)

    training_data = []
    ### Train Model
    my_model = models.random_model()
    my_model.train(training_data)


    test_data = range(0,2000)
    ### Test Model
    predictions = my_model.predict(test_data)


    ### Save predictions to file
    ### prediction is a list of values
    def write_file (predictions):
        filename = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".csv"
        os.makedirs("output", exist_ok=True)

        with open("output/" + filename, "w") as file:
            file.write("Id,Prediction")
            for i in range(0, len(predictions)):
                # save random predictions as baseline. Should be about 50% correct
                file.write("\n" + str(i + 1) + "," + predictions[i])
                print("output written to file: output/" + filename)

    write_file(predictions)          
  
    
   

if __name__ == '__main__':
    main()
