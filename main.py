#!/usr/bin/python3
import models
import helper
import feature_extraction

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics

def main ():

    ### Load training data
    print("data loading...")
    training_data = helper.read_file("./data/train.txt")
    print("data loaded")
    
    #### Extract features
    numbers = feature_extraction.get_followers(training_data)
    print("features extracted")
    numbers.to_csv("data/feature_number.csv")
    print("features written to file ")

    ### Train Model
    model_f = models.friends_model()
    model_r = models.random_model()
    model_f.train(numbers)
    model_r.train(numbers)


    ### Test Model
    filename = "data/generated-test-data.test"
    test_data =pd.read_csv(filename, sep='\t', lineterminator='\r', index_col= "Id")
    test_data.loc[:1000,"Label"] = True
    test_data.loc[1000:,"Label"] = False
        
    ### Evaluate Model 
    #friends model
    predictions = model_f.predict(test_data["Source"])
    auc = sklearn.metrics.roc_auc_score(test_data["Label"],predictions)
    print("AUC friends: ", auc)
    
    #random model
    predictions = model_r.predict(test_data["Source"])
    auc = sklearn.metrics.roc_auc_score(test_data["Label"],predictions)
    print("AUC random: ", auc)


if __name__ == '__main__':
    main()