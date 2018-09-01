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
    following_dict = helper.read_file("./data/train.txt")
    followers_dict = helper.read_file("./data/followers.txt")
    print("data loaded")

    ### todo: to Jonathan: you need to outcomment below and run to generate followers.txt file
    # followers_dict = helper.extract_followers_from_following(following_dict)
    # helper.write_dict_with_lists_to_file(followers_dict, "followers.txt")

    print(len(following_dict))
    print(len(followers_dict))



    #### Extract features
    print("extracting features...")



    print("features extracted")



    # ### Train Model
    # model_f = models.friends_model()
    # model_r = models.random_model()
    # model_f.train(user_followers_dict)
    # model_r.train(None)
    #
    #
    #
    #
    # ### Test Model
    # filename = "data/generated-test-data.test"
    # test_data =pd.read_csv(filename, sep='\t', lineterminator='\n', index_col= "Id")
    # test_data.loc[:1000,"Label"] = True
    # test_data.loc[1000:,"Label"] = False
    #
    # ### Evaluate Model
    # #friends model
    # predictions = model_f.predict(test_data["Source"])
    # auc = sklearn.metrics.roc_auc_score(test_data["Label"],predictions)
    # print("AUC friends: ", auc)
    #
    # #random model
    # predictions = model_r.predict(test_data["Source"])
    # auc = sklearn.metrics.roc_auc_score(test_data["Label"],predictions)
    # print("AUC random: ", auc)


if __name__ == '__main__':
    main()