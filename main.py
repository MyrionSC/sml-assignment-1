#!/usr/bin/python3
import models
import helper

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics

def main ():
    ### Inits
    # vertex_set = set()
    # sink_dict = {}

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

    ### Test Model
    filename = "data/generated-test-data.test"
    test_data =pd.read_csv(filename, sep='\t', lineterminator='\r', index_col= "Id")
    test_data.loc[:1000,"Label"] = True
    test_data.loc[1000:,"Label"] = False
        
    ### Evaluate Model 
    predictions = my_model.predict(test_data)
    auc = sklearn.metrics.roc_auc_score(test_data["Label"],predictions)
    print("AUC:", auc)


if __name__ == '__main__':
    main()