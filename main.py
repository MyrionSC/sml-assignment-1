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