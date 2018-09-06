import pandas as pd
import math
import helper
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


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
    following_dict = helper.read_file_to_dict("./data/train.txt")
    followers_dict = helper.read_file_to_dict("./data/followers.txt")
    print("data prepared")
    print()
    print()


    # num of source / sinks in train.txt
    print("num of Source and sinks that we have following data for")
    print("gen-data source: " + str(sum(data_df.Source.apply(lambda r: 1 if r in following_dict else 0))))
    print("gen-data sink: " + str(sum(data_df.Sink.apply(lambda r: 1 if r in following_dict else 0))))
    print("test-public source: " + str(sum(test_df.Source.apply(lambda r: 1 if r in following_dict else 0))))
    print("test-public sink: " + str(sum(test_df.Sink.apply(lambda r: 1 if r in following_dict else 0))))
    print()

    # plot normal distribution of following numbers
    # print("Normal distribution of number of followers: (format: TITLE: MEAN, STANDARD_DEVIATION)")
    # plot_normal_distribution("gen-data source following", data_df.Source, following_dict)
    # plot_normal_distribution("gen-data sink following", data_df.Sink, following_dict)
    # plot_normal_distribution("public-test source following", test_df.Source, following_dict)
    # plot_normal_distribution("public-test sink following", test_df.Sink, following_dict)
    # print()
    # print()

    print("*** Hist plots")
    plot_histogram("gen-data sink true following num histogram", data_df.Sink[data_df.Label == True], following_dict)
    plot_histogram("gen-data sink false following num histogram", data_df.Sink[data_df.Label == False], following_dict)
    plot_histogram("gen-data sink both following num histogram", data_df.Sink, following_dict)
    plot_histogram("test-public sink following num histogram", test_df.Source, following_dict)
    print()
    plot_histogram("gen-data sink true followers num histogram", data_df.Sink[data_df.Label == True], followers_dict)
    plot_histogram("gen-data sink false followers num histogram", data_df.Sink[data_df.Label == False], followers_dict)
    plot_histogram("gen-data sink both followers num histogram", data_df.Sink, followers_dict)
    plot_histogram("test-public sink followers num histogram", test_df.Source, followers_dict)
    print()
    # plot_histogram("gen-data source following num histogram", data_df.Source, following_dict)
    # plot_histogram("gen-data sink following num histogram", data_df.Sink, following_dict)
    # plot_histogram("test-public source following num histogram", test_df.Source, following_dict)
    # plot_histogram("test-public sink following num histogram", test_df.Sink, following_dict)
    # print()
    # plot_histogram("gen-data source followers num histogram", data_df.Source, followers_dict)
    # plot_histogram("gen-data sink followers num histogram", data_df.Sink, followers_dict)
    # plot_histogram("test-public source followers num histogram", test_df.Source, followers_dict)
    # plot_histogram("test-public sink followers num histogram", test_df.Sink, followers_dict)
    print()
    print()

    ### todo: to Jonathan: you need to outcomment 2 lines below and run to generate followers.txt file
    # followers_dict = helper.extract_followers_from_following(following_dict)
    # helper.write_dict_with_lists_to_file(followers_dict, "followers.txt")

    print("Average (format: Following_num_avg Followers_num_avg)")
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



def plot_normal_distribution(title, data_series, dict):
    l = data_series.apply(lambda s: len(dict[s]) if s in dict else 1).values

    mu, std = norm.fit(l)
    print(title + ": " + str(mu) + ", " + str(std))
    x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    plt.plot(x, norm.pdf(x, mu, std))
    plt.title(title)
    plt.show()

def plot_histogram(title, data_series, dict):
    x = []
    for key in data_series.values:
        # if key in dict and len(dict[key]) < 20000:
        if key in dict:
            x.append(len(dict[key]))

    print(title)
    print("count: " + str(len(x)))
    print("mean: " + str(sum(x) / len(x)))
    print()

    # x = np.random.normal(size=1000)
    plt.title(title)
    # bins = 10 ** (np.arange(0, 3, 0.10)) # do it yourself x axis log
    bins = (np.arange(0, 100))

    plt.hist(x, log=True, bins=bins)
    print(plt.axis())
    # x1, x2, y1, y2 = plt.axis()
    # plt.axis((x1,x2,y1,200))
    plt.show()

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
