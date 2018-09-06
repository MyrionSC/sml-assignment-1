#!/usr/bin/python3
import models
import helper
import feature_extraction

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def extract_features(df, followers_dict, following_dict):
    #buidling vectorized functions:
    vec_following_followers = np.vectorize(feature_extraction.same_following_followers)
    vec_followers_followings = np.vectorize(feature_extraction.same_followers_following)

    df["Reciprocated"] = df.apply(lambda row: feature_extraction.reciprocated_follows(row["Source"], row["Sink"], followers_dict), axis=1)
    df["Same_Follows"] = df.apply(lambda row: feature_extraction.same_following(row["Source"], row["Sink"], following_dict), axis=1)
    df["Same_Followers"] = df.apply(lambda row: feature_extraction.same_followers(row["Source"], row["Sink"], followers_dict), axis=1)
    df["following_followers"] = vec_following_followers(df["Source"], df["Sink"], followers_dict, following_dict)
    df["followers_following"] = vec_followers_followings(df["Source"], df["Sink"], followers_dict, following_dict)
    return df

def train(train_df,feature_cols):
    print("Training model...")
    # feature_cols = ["Source_Following", "Source_Followers", "Sink_Following", "Sink_Followers",
    #                 "Reciprocated", "Same_Follows", "Same_Followers"]
    features = train_df.loc[:, feature_cols]
    target = train_df.Label

    # try XGBRegressor
    # try predict_proba
    # try XGBRanker
    model = XGBClassifier()
    model.fit(features.values, target.values)
    print("Model trained")
    return model

def build_features(data_path,following_dict_path,followers_dict_path):
    ### Load training data
    print("Training and test data loading...")
    data_df = pd.read_csv(data_path, sep="\t", index_col="Id")
    print("Training and test data loaded")

    ### load feature extraction helper data
    print("Feature data loading...")
    following_dict = helper.read_file_to_dict(following_dict_path)
    followers_dict = helper.read_file_to_dict(followers_dict_path)
    print("Feature data loaded")
    return extract_features(data_df,following_dict,followers_dict)



def main ():
    # Path:     
    data= "data/generated-data.txt"
    following_dict = "./data/train.txt"
    followers_dict = "./data/followers.txt"
    feature_cols = ["Reciprocated", "Same_Follows", "Same_Followers", "followers_following","following_followers"]
    test_public_prediction = False
        
    print("Extracting features...")
    data_df = build_features(data,following_dict, followers_dict) 
    
    if test_public_prediction:
        train_df = data_df.copy()
        test_df = build_features("data/test-public.txt",following_dict, followers_dict)
        test_public_prediction = True
    else:
        train_df, test_df = train_test_split(data_df, test_size=0.2)
        train_df = train_df.copy()
        test_df = test_df.copy()
    
    print("Features extracted")
    model = train(train_df,feature_cols)
    print("Model trained")

    ### Test model
    test_features = test_df.loc[:, feature_cols]
    predictions = model.predict_proba(test_features.values)

    if test_public_prediction:
        helper.save_predictions_to_file(test_features, predictions)
    else:
        auc = sklearn.metrics.roc_auc_score(test_df["Label"],predictions)
        print("AUC: ", auc)


if __name__ == '__main__':
    main()