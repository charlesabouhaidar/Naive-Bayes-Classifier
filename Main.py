import matplotlib
import math
import numpy as np
import pandas as pd
import nltk
import sys

# -------------------------------------------------------
# Assignment 2
# Written by Charles Abou Haidar, 40024373
# For COMP 472 – Summer 2020
# --------------------------------------------------------

data_set = pd.read_csv("hns_2018_2019.csv") # open csv file
data_frame = pd.DataFrame(data_set) # transform csv file to dataframe
df_2018 = data_frame.loc[data_frame['Created At'].str.contains("2018")] # extract data where 'Created At' = 2018
testing_data_set_2019 = data_frame.loc[data_frame['Created At'].str.contains("2019")] # testing data set (2019)
lower_case_df = df_2018['Title'].str.lower().str.split() # change all vocab to lower case and split
vocab = []
# special_characters = [':', ',', ';', '!', '.', '?', '/', ']', '[', '+', '-', '{', '}', '-', '_', '`', '~', '@', '#', '$', '%', '^', '&', '*', '(', ')']
for row in lower_case_df:
    for word in row:
        vocab.append(word)


print(vocab)
"""
To filter all words in vocab
vocab = set()
df['Title'].str.lower().str.split().apply(vocab.update)
"""

# create vocabulary list which has all the words contained in Title
vocabulary = []

# compute frequency of word in the df

# calculate probability of word in vocabulary for each post type

# lowercase every word from Title column, tokenize them and use them as vocabulary

# TO DO BEFORE WEEKEND
