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


def extract_features(df, followers_dict, following_dict):
    df = df.copy()
    #buidling vectorized functions:
    vec_following_followers = np.vectorize(feature_extraction.same_following_followers)
    vec_followers_followings = np.vectorize(feature_extraction.same_followers_following)

    df["Jacard_similarity"] = df.apply(lambda row: feature_extraction.jacard_similarity(row["Source"], row["Sink"], following_dict, followers_dict), axis=1)
    #df["Reciprocated"] = df.apply(lambda row: feature_extraction.reciprocated_follows(row["Source"], row["Sink"], followers_dict), axis=1)
    #df["Same_Follows"] = df.apply(lambda row: feature_extraction.same_following(row["Source"], row["Sink"], following_dict), axis=1)
    #df["Same_Followers"] = df.apply(lambda row: feature_extraction.same_followers(row["Source"], row["Sink"], followers_dict), axis=1)
    #df["Following_followers"] = vec_following_followers(df["Source"], df["Sink"], followers_dict, following_dict)
    #df["Followers_following"] = vec_followers_followings(df["Source"], df["Sink"], followers_dict, following_dict)
    #df["#follower_Source"] = df.apply(lambda row: feature_extraction.dict_value_len(row["Source"],  followers_dict), axis=1)
    #df["#following_Source"] = df.apply(lambda row: feature_extraction.dict_value_len(row["Source"],  following_dict), axis=1)
    #df["#follower_Sink"] = df.apply(lambda row: feature_extraction.dict_value_len(row["Sink"], followers_dict), axis=1)
    #df["#following_Sink"] = df.apply(lambda row: feature_extraction.dict_value_len(row["Sink"], following_dict), axis=1)


    return df

def normalize_jac(jac,scale= 2.3):
    jac_loc =np.log(np.round(jac,4) *10000)
    jac_loc = jac_loc.replace([np.inf, -np.inf], 0)
    df = jac_loc.to_frame()
    df["1"] = 10
    df.iloc[:,0] = df.iloc[:,0]*scale
    res = df.min(axis=1)/10
    return res


def main ():
    test_public_prediction = True
    start_time = time.time()
    data_file = "data/generated-data-new-2.txt"
    following_file = "./data/train.txt"
    followers_file = "./data/followers.txt"

    data_df,following_dict, followers_dict = load_data(data_file,following_file,followers_file)

    ### Extract features
    if test_public_prediction:
        train_df = data_df.copy()
        test_df = pd.read_csv("data/test-public.txt", sep="\t", index_col="Id")
    else:
        train_df, test_df = train_test_split(data_df, test_size=0.2)
        train_df, test_df = train_df.copy(), test_df.copy()

    print("Extracting features...")
    #train_df = extract_features(train_df,followers_dict, following_dict)
    test_df = extract_features(test_df,followers_dict, following_dict)
    print("Features extracted")

    # save features to file
    # feature_df = pd.concat([train_df, test_df])[['Label','Jacard_similarity']]
    feature_df = pd.concat([train_df, test_df])
    feature_df.to_csv("features_gen_new.csv")

    ### Train model
    print("Training model...")
    feature_cols = ["Jacard_similarity"]#, "Reciprocated", "Same_Follows", "Same_Followers", "Followers_following","Following_followers"]
    
    features = train_df.loc[:, feature_cols]
    target = train_df.Label
    
    

    modelXGB = XGBClassifier()
    modelXGB.fit(features.values, target.values)
    print("Model trained")

    logreg_model = LogisticRegression()
    logreg_model.fit(features, target)


    ### Test model
    test_features = test_df.loc[:, feature_cols]
    predictions_XGB = modelXGB.predict(test_features.values)
    predictions_logreg = logreg_model.predict(test_features)


    # predictions_p = model.predict_proba(test_features.values)
    # predictions = predictions_p[:,1].tolist() # column 1 is the true prediction
    predictions = normalize_jac(test_df["Jacard_similarity"])

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