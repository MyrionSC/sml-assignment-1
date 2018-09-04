#!/usr/bin/python3
import helper
import feature_extraction

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main ():
    test_public_prediction = False

    ### Load training data
    print("Training and test data loading...")
    data_df = pd.read_csv("data/generated-data.txt", sep="\t", index_col="Id")
    print("Training and test data loaded")

    ### load feature extraction helper data
    print("Feature data loading...")
    following_dict = helper.read_file_to_dict("./data/train.txt")
    followers_dict = helper.read_file_to_dict("./data/followers.txt")
    print("Feature data loaded")


    ### Extract features
    if test_public_prediction:
        train_df = data_df.copy()
        test_df = pd.read_csv("data/test-public.txt", sep="\t", index_col="Id")
        test_public_prediction = True
    else:
        train_df, test_df = train_test_split(data_df, test_size=0.2)
        train_df = train_df.copy()
        test_df = test_df.copy()

    print("Extracting features...")
    #buidling vectorized functions:
    vec_following_followers = np.vectorize(feature_extraction.same_following_followers)
    vec_followers_followings = np.vectorize(feature_extraction.same_followers_following)

    train_df["Reciprocated"] = train_df.apply(lambda row: feature_extraction.reciprocated_follows(row["Source"], row["Sink"], followers_dict), axis=1)
    train_df["Same_Follows"] = train_df.apply(lambda row: feature_extraction.same_following(row["Source"], row["Sink"], following_dict), axis=1)
    train_df["Same_Followers"] = train_df.apply(lambda row: feature_extraction.same_followers(row["Source"], row["Sink"], followers_dict), axis=1)
    train_df["Following_followers"] = vec_following_followers(train_df["Source"], train_df["Sink"], followers_dict, following_dict)
    train_df["Followers_following"] = vec_followers_followings(train_df["Source"], train_df["Sink"], followers_dict, following_dict)
    # train_df["Following_following_follower"] = train_df.apply(lambda row: feature_extraction.following_following_follower(row["Source"], row["Sink"], following_dict, followers_dict), axis=1)

    
    test_df["Reciprocated"] = test_df.apply(lambda row: feature_extraction.reciprocated_follows(row["Source"], row["Sink"], followers_dict), axis=1)
    test_df["Same_Follows"] = test_df.apply(lambda row: feature_extraction.same_following(row["Source"], row["Sink"], following_dict), axis=1)
    test_df["Same_Followers"] = test_df.apply(lambda row: feature_extraction.same_followers(row["Source"], row["Sink"], followers_dict), axis=1)
    test_df["Following_followers"] = vec_following_followers(test_df["Source"], test_df["Sink"], followers_dict, following_dict)
    test_df["Followers_following"] = vec_followers_followings(test_df["Source"], test_df["Sink"], followers_dict, following_dict)
    # test_df["Following_following_follower"] = test_df.apply(lambda row: feature_extraction.following_following_follower(row["Source"], row["Sink"], following_dict, followers_dict), axis=1)
    print("Features extracted")


    ### Train model
    print("Training model...")
    feature_cols = ["Reciprocated", "Same_Follows", "Same_Followers", "Followers_following","Following_followers"]
    features = train_df.loc[:, feature_cols]
    target = train_df.Label

    model = XGBClassifier()
    model.fit(features.values, target.values)
    print("Model trained")

    # feature_df = pd.concat([train_df, test_df])
    # feature_df.to_csv("features.csv")

    ### Test model
    test_features = test_df.loc[:, feature_cols]
    predictions = model.predict(test_features.values)
    # predictions_p = model.predict_proba(test_features.values)
    # predictions = predictions_p[:,1].tolist() # column 1 is the true prediction

    if test_public_prediction:
        helper.save_predictions_to_file(test_features, predictions)
    else:
        auc = sklearn.metrics.roc_auc_score(test_df["Label"],predictions)
        print("AUC: ", auc)


if __name__ == '__main__':
    main()