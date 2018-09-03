import pandas as pd
import helper


'''
I tried to look at the average following and followers number of the source and sink nodes of the edges in various data sets

Results after doing a couple of runs with different edge data sets are below

format: Following_num_avg Followers_num_avg

- Fake edges
Source: 1099.9331: 91.6845
Sink: 1.8724: 4.7324

- Real edges
Source: 1129.3976: 91.7012
Sink: 2253.7726: 187.7634

- Real and Fake edges
Source: 1114.66535: 91.69285
Sink: 1127.8225: 96.2479

- Test-Public
Source: 1329.491: 90.922
Sink: 1820.5505: 143.222


Comment: We should try to make the Real and Fake edges result look somewhat similar to the Test-Public result for our generated data

'''

def main ():

    ### Load training data
    print("Prepairing data...")
    data_df = pd.read_csv("data/generated-data.txt", sep="\t", index_col="Id")
    test_df = pd.read_csv("data/test-public.txt", sep="\t", index_col="Id")

    ### load feature extraction helper data
    following_dict = helper.read_file("./data/train.txt")
    followers_dict = helper.read_file("./data/followers.txt")
    print("data prepared")
    print()
    print()

    ### todo: to Jonathan: you need to outcomment 2 lines below and run to generate followers.txt file
    # followers_dict = helper.extract_followers_from_following(following_dict)
    # helper.write_dict_with_lists_to_file(followers_dict, "followers.txt")

    print("format: Following_num_avg Followers_num_avg")
    print()

    print("Real edges")
    generated_real_edges_df = data_df[data_df.Label == True]
    extract_followers_following_avg(generated_real_edges_df, following_dict, followers_dict)
    print()

    print("Fake edges")
    generated_fake_edges_df = data_df[data_df.Label == False]
    extract_followers_following_avg(generated_fake_edges_df, following_dict, followers_dict)
    print()

    print("Real and fake edges")
    generated_edges_df = data_df
    extract_followers_following_avg(generated_edges_df, following_dict, followers_dict)
    print()

    print("Test-Public")
    extract_followers_following_avg(test_df, following_dict, followers_dict)


def extract_followers_following_avg(edges_df, following_dict, followers_dict):
    data_source_following_sum = 0
    data_source_followers_sum = 0
    data_sink_following_sum = 0
    data_sink_followers_sum = 0
    for index, row in edges_df.iterrows():
        data_source_following_sum += len(following_dict[row["Source"]])
        data_source_followers_sum += len(followers_dict[row["Source"]])
        if row["Sink"] in following_dict:
            data_sink_following_sum += len(following_dict[row["Sink"]])
        data_sink_followers_sum += len(followers_dict[row["Sink"]])

    data_source_following_avg = data_source_following_sum / len(edges_df)
    data_source_followers_avg = data_source_followers_sum / len(edges_df)
    data_sink_following_avg = data_sink_following_sum / len(edges_df)
    data_sink_followers_avg = data_sink_followers_sum / len(edges_df)
    print("Source: " + str(data_source_following_avg) + ": " + str(data_source_followers_avg))
    print("Sink: " + str(data_sink_following_avg) + ": " + str(data_sink_followers_avg))


if __name__ == '__main__':
    main()
