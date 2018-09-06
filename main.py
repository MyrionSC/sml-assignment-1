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

def load_data(data,following,followers):
    ### Load training data
    print("Training and test data loading...")
    data_df = pd.read_csv(data, sep="\t", index_col="Id")
    print("Training and test data loaded")

    ### load feature extraction helper data
    print("Feature data loading...")
    following_dict = helper.read_file_to_dict(following)
    followers_dict = helper.read_file_to_dict(followers)
    print("Feature data loaded")
    return data_df,following_dict,followers_dict

def main ():
    data_file = "data/generated-data.txt"
    following_file = "./data/train.txt"
    followers_file = "./data/followers.txt"

    _, following_dict, followers_dict = load_data(data_file,following_file,followers_file)
    test_df = pd.read_csv("data/test-public.txt", sep="\t", index_col="Id")

    test_df["Jacard_similarity"] = test_df.apply(lambda row: feature_extraction.jaccard_similarity(row["Source"], row["Sink"], following_dict, followers_dict), axis=1)
    jaccard_max = test_df["Jacard_similarity"].max()
    norm = 1 / jaccard_max
    test_df["Prediction"] = test_df["Jacard_similarity"] * norm
    prediction_df = test_df.drop(["Source", "Sink", "Jacard_similarity"], axis=1)
    prediction_df.to_csv("jaccard-index-prediction.csv")

if __name__ == '__main__':
    main()