#!/usr/bin/python3
import models
import helper
import feature_extraction

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics
from sklearn.linear_model import LogisticRegression

def main ():


    ### Load training data
    print("Training and test data loading...")
    train_df = pd.read_csv("data/generated-training-data.test", sep="\t", index_col="Id")
    test_df = pd.read_csv("data/generated-test-data.test", sep="\t", index_col="Id")
    print("Training and test data loaded")

    # apply labels to training and test data
    train_df.loc[:9000, "Label"] = True
    train_df.loc[9000:, "Label"] = False
    test_df.loc[:1000, "Label"] = True
    test_df.loc[1000:, "Label"] = False


    ### load feature extraction helper data
    print("Feature data loading...")
    following_dict = helper.read_file("./data/train.txt")
    followers_dict = helper.read_file("./data/followers.txt")

    ### todo: to Jonathan: you need to outcomment 2 lines below and run to generate followers.txt file
    # followers_dict = helper.extract_followers_from_following(following_dict)
    # helper.write_dict_with_lists_to_file(followers_dict, "followers.txt")
    print("Feature data loaded")


    ### Extract features
    print("Extracting features...")
    train_df["Reciprocated"] = train_df.apply(lambda row: feature_extraction.reciprocated_follows(row["Source"], row["Sink"],
                                                                                          followers_dict), axis=1)
    test_df["Reciprocated"] = test_df.apply(lambda row: feature_extraction.reciprocated_follows(row["Source"], row["Sink"],
                                                                                          followers_dict), axis=1)
    train_df["Same_Follows"] = train_df.apply(lambda row: feature_extraction.same_following(row["Source"], row["Sink"],
                                                                                          following_dict), axis=1)
    test_df["Same_Follows"] = test_df.apply(lambda row: feature_extraction.same_following(row["Source"], row["Sink"],
                                                                                          following_dict), axis=1)
    train_df["Same_Followers"] = train_df.apply(lambda row: feature_extraction.same_followers(row["Source"], row["Sink"],
                                                                                          followers_dict), axis=1)
    test_df["Same_Followers"] = test_df.apply(lambda row: feature_extraction.same_followers(row["Source"], row["Sink"],
                                                                                          followers_dict), axis=1)
    print("Features extracted")


    ### Train model
    print("Training model...")
    feature_cols = ["Reciprocated", "Same_Follows", "Same_Followers"]
    features = train_df.loc[:, feature_cols]
    target = train_df.Label

    # trying logreg for new
    logreg = LogisticRegression()
    logreg.fit(features, target)
    print("Model trained")


    ### Test model
    test_features = test_df.loc[:, feature_cols]
    predictions = logreg.predict(test_features)


    ### Evaluate Model
    auc = sklearn.metrics.roc_auc_score(test_df["Label"],predictions)
    print("AUC friends: ", auc)


if __name__ == '__main__':
    main()