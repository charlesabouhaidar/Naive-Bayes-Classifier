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

data_set = pd.read_csv("hns_2018_2019.csv")
df = pd.DataFrame(data_set, columns=['Title', 'Created At'])

# need to filter the df to the rows which have 2018 as the first 4 digits of the column 'Created At'

# create vocabulary list which has all the words contained in Title
vocabulary = []

# compute frequency of word in the df

# calculate probability of word in vocabulary for each post type

# lowercase every word from Title column, tokenize them and use them as vocabulary

# TO DO BEFORE WEEKEND
