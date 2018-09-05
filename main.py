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

def main ():
    test_public_prediction = False
    start_time = time.time()

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
    else:
        train_df, test_df = train_test_split(data_df, test_size=0.2)
        train_df, test_df = train_df.copy(), test_df.copy()

    print("Extracting features...")


    #buidling vectorized functions:
    vec_following_followers = np.vectorize(feature_extraction.same_following_followers)
    vec_followers_followings = np.vectorize(feature_extraction.same_followers_following)

    # train_df["Sink_following"] = train_df.apply(lambda row: feature_extraction.dict_value_len(row["Sink"], following_dict), axis=1)
    # train_df["Sink_followers"] = train_df.apply(lambda row: len(followers_dict[row["Sink"]]), axis=1)
    train_df["Jacard_similarity"] = train_df.apply(lambda row: feature_extraction.jacard_similarity(row["Source"], row["Sink"], following_dict, followers_dict), axis=1)
    train_df["Reciprocated"] = train_df.apply(lambda row: feature_extraction.reciprocated_follows(row["Source"], row["Sink"], followers_dict), axis=1)
    train_df["Same_Follows"] = train_df.apply(lambda row: feature_extraction.same_following(row["Source"], row["Sink"], following_dict), axis=1)
    train_df["Same_Followers"] = train_df.apply(lambda row: feature_extraction.same_followers(row["Source"], row["Sink"], followers_dict), axis=1)
    train_df["Following_followers"] = vec_following_followers(train_df["Source"], train_df["Sink"], followers_dict, following_dict)
    train_df["Followers_following"] = vec_followers_followings(train_df["Source"], train_df["Sink"], followers_dict, following_dict)
    # test_df["Src_following_following_sink"] = test_df.apply(lambda row: feature_extraction.following_following_follower(row["Source"], row["Sink"], following_dict, followers_dict), axis=1)
    # test_df["Sink_following_following_src"] = test_df.apply(lambda row: feature_extraction.following_following_follower(row["Sink"], row["Source"], following_dict, followers_dict), axis=1)


    # test_df["Sink_following"] = test_df.apply(lambda row: feature_extraction.dict_value_len(row["Sink"], following_dict), axis=1)
    # test_df["Sink_followers"] = test_df.apply(lambda row: len(followers_dict[row["Sink"]]), axis=1)
    test_df["Jacard_similarity"] = test_df.apply(lambda row: feature_extraction.jacard_similarity(row["Source"], row["Sink"], following_dict, followers_dict), axis=1)
    test_df["Reciprocated"] = test_df.apply(lambda row: feature_extraction.reciprocated_follows(row["Source"], row["Sink"], followers_dict), axis=1)
    test_df["Same_Follows"] = test_df.apply(lambda row: feature_extraction.same_following(row["Source"], row["Sink"], following_dict), axis=1)
    test_df["Same_Followers"] = test_df.apply(lambda row: feature_extraction.same_followers(row["Source"], row["Sink"], followers_dict), axis=1)
    test_df["Following_followers"] = vec_following_followers(test_df["Source"], test_df["Sink"], followers_dict, following_dict)
    test_df["Followers_following"] = vec_followers_followings(test_df["Source"], test_df["Sink"], followers_dict, following_dict)
    # test_df["Src_following_following_sink"] = test_df.apply(lambda row: feature_extraction.following_following_follower(row["Source"], row["Sink"], following_dict, followers_dict), axis=1)
    # test_df["Sink_following_following_src"] = test_df.apply(lambda row: feature_extraction.following_following_follower(row["Sink"], row["Source"], following_dict, followers_dict), axis=1)
    print("Features extracted")

    # save features to file
    feature_df = pd.concat([train_df, test_df])[['Label','Jacard_similarity']]
    feature_df.to_csv("features.csv")

    ### Train model
    print("Training model...")
    feature_cols = ["Reciprocated", "Same_Follows", "Same_Followers", "Followers_following","Following_followers"]
    features = train_df.loc[:, feature_cols]
    target = train_df.Label

    model = XGBClassifier()
    model.fit(features.values, target.values)
    print("Model trained")

    logreg_model = LogisticRegression()
    logreg_model.fit(features, target)


    ### Test model
    test_features = test_df.loc[:, feature_cols]
    predictions = model.predict(test_features.values)
    predictions_logreg = logreg_model.predict(test_features)


    # predictions_p = model.predict_proba(test_features.values)
    # predictions = predictions_p[:,1].tolist() # column 1 is the true prediction

    if test_public_prediction:
        helper.save_predictions_to_file(test_features, predictions)
    else:
        auc = sklearn.metrics.roc_auc_score(test_df["Label"],predictions)
        auc_logreg = sklearn.metrics.roc_auc_score(test_df["Label"],predictions_logreg)
        print("AUC: ", auc)
        print("AUC_logreg: ", auc_logreg)

    print()
    print("execution time: " + str(time.time() - start_time))

if __name__ == '__main__':
    main()