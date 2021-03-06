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

def load_data(following):
    ### load feature extraction helper data
    print("Feature data loading...")
    following_dict = helper.read_file_to_dict(following)
    followers_dict = helper.extract_followers_from_following(following_dict)
    print("Feature data loaded")
    return following_dict,followers_dict


def normalize_jac(jac,scale= 2.6):
    jac_loc =np.log(np.round(jac,4) *10000)
    jac_loc = jac_loc.replace([np.inf, -np.inf], 0)
    df = jac_loc.to_frame()
    df["1"] = 10
    df.iloc[:,0] = df.iloc[:,0]*scale
    res = df.min(axis=1)/10
    return res


def main ():
    following_file = "./data/train.txt"
    prediction_file = "prediction.csv"

    following_dict, followers_dict = load_data(following_file)
    test_df = pd.read_csv("data/test-public.txt", sep="\t", index_col="Id")

    print("creating features")
    test_df["Jacard_similarity"] = test_df.apply(lambda row: feature_extraction.jaccard_similarity(row["Source"], row["Sink"], following_dict, followers_dict), axis=1)
    test_df["Prediction"] = normalize_jac(test_df["Jacard_similarity"])
    prediction_df = test_df.drop(["Source", "Sink", "Jacard_similarity"], axis=1)
    prediction_df.to_csv(prediction_file)
    print("predictions written to file: " + prediction_file)

if __name__ == '__main__':
    main()