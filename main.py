#!/usr/bin/python3
import helper
import feature_extraction

import time
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

    df["Jacard_similarity"] = df.apply(lambda row: feature_extraction.jacard_similarity(row["Source"], row["Sink"], following_dict, followers_dict), axis=1)
    df["Reciprocated"] = df.apply(lambda row: feature_extraction.reciprocated_follows(row["Source"], row["Sink"], followers_dict), axis=1)
    df["Same_Follows"] = df.apply(lambda row: feature_extraction.same_following(row["Source"], row["Sink"], following_dict), axis=1)
    df["Same_Followers"] = df.apply(lambda row: feature_extraction.same_followers(row["Source"], row["Sink"], followers_dict), axis=1)
    df["following_followers"] = vec_following_followers(df["Source"], df["Sink"], followers_dict, following_dict)
    df["followers_following"] = vec_followers_followings(df["Source"], df["Sink"], followers_dict, following_dict)
    return df


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
    start_time = time.time()
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
    
    feature_cols = ["Jacard_similarity", "Reciprocated", "Same_Follows", "Same_Followers", "Followers_following","Following_followers"]
    
    features = train_df.loc[:, feature_cols]
    target = train_df.Label

    model = XGBClassifier()
    model.fit(features.values, target.values)
    print("XGB trained")

    logreg_model = LogisticRegression()
    logreg_model.fit(features, target)
    predictions_logreg = logreg_model.predict(test_df[feature_cols])
    print("logreg trained")

    ### Test model
    predictions = model.predict_proba(test_df[feature_cols].values)

    if test_public_prediction:
        helper.save_predictions_to_file(test_df[feature_cols].values, predictions)
    else:
        auc = sklearn.metrics.roc_auc_score(test_df["Label"],predictions)
        auc_logreg = sklearn.metrics.roc_auc_score(test_df["Label"],predictions_logreg)
        print("AUC: ", auc)
        print("AUC_logreg: ", auc_logreg)

    print("execution time: " + str(time.time() - start_time))

if __name__ == '__main__':
    main()