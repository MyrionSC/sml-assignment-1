File structure of zip-file:

main.py
helper.py
feature-extraction.py
data-generator.py
data-analysis.py
data/


A prediction file called prediction.csv is generated by running main.py with python3

As requested, no data files are included in the zipped file. For the prediction to work, train.txt and public-test.txt must be in data directory.

data-analysis.py is not required for the final prediction, but were written to provide an understanding of how training data should be generated, and is provided as referencing material.
data-generator.py generates training data for a supervised learning approach to answer the problem. In the end we went with an unsupervised approach so the generated data does not play into the final model, but it is added for referencing purposes.





