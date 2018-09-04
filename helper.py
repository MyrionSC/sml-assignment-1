import os
from datetime import datetime
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def read_file_to_dict (filename):
    data_dict = dict()
    with open(filename) as trainfile:
        for line in trainfile:
            line_list = [int(k) for k in line.split("\t")]
            data_dict[line_list[0]] = line_list[1:]
    return data_dict



    ### Save predictions to file
    ### prediction is a list of values
    # write_file(predictions)

def save_predictions_to_file (test_features, predictions):
    filename = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".csv"
    os.makedirs("output", exist_ok=True)

    predictions_df = pd.DataFrame({'Id': test_features.index, 'Prediction': list(map(int, predictions))}).set_index('Id')
    predictions_df.to_csv('output/' + filename)

    print("output written to file: output/" + filename)


def write_dict_with_lists_to_file (list_dict, filename):
    os.makedirs("data", exist_ok=True)

    with open("data/" + filename, "w+") as file:
        for key, value in list_dict.items():
            file.write(str(key))
            for item in value:
                file.write("\t" + str(item))
            file.write("\n")

def extract_followers_from_following(followers_dict):
    following_dict = dict()

    for user, followers in followers_dict.items():
        for f in followers:
            if f in following_dict:
                following_dict[f].append(user)
            else:
                following_dict[f] = [user]

    return following_dict
