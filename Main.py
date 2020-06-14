import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import nltk
import sys
import time
import csv
# -------------------------------------------------------
# Assignment 2
# Written by Charles Abou Haidar, 40024373
# For COMP 472 – Summer 2020
# --------------------------------------------------------

SMOOTHING_VALUE = 0.5

start_time = time.time() #start time
reader = pd.read_csv("hns_2018_2019.csv") #open csv file
data_frame = pd.DataFrame(reader) # convert csv file to data frame
training_set = data_frame.loc[data_frame['Created At'].str.contains("2018")] # get all info from dataframe where the year is 2018 for training set
testing_set = data_frame.loc[data_frame['Created At'].str.contains("2019")] # get all the info from dataframe where the year is 2019 for testings et

# extracting all the title text from all post types and converting them to a list
story = training_set['Title'].loc[training_set['Post Type'] == 'story'].values.tolist() # get all the story post types in 2018
ask = training_set['Title'].loc[training_set['Post Type'] == 'ask_hn'].values.tolist() # get all the ask_hn post types in 2018
show = training_set['Title'].loc[training_set['Post Type'] == 'show_hn'].values.tolist() # get all the show_hn post types in 2018
poll = training_set['Title'].loc[training_set['Post Type'] == 'poll'].values.tolist() # get all the poll post types in 2018

# converting all post type Titles into lower case
story = [element.lower() for element in story]
ask = [element.lower() for element in ask]
show = [element.lower() for element in show]
poll = [element.lower() for element in poll]

# splitting all elements in list of Title post types into separate words delimited by whitespace
story_list = [word for line in story for word in line.split()] # split list elements by whitespace into words for story list
ask_list = [word for line in ask for word in line.split()] # split list elements by whitespace into words for ask_hn list
show_list = [word for line in show for word in line.split()] # split list elements by whitespace into words for show_hn list
poll_list = [word for line in poll for word in line.split()] # split list elements by whitespace into words for poll list

vocabulary = story_list + ask_list + show_list + poll_list
vocabulary_list = [word for line in vocabulary for word in line.split()]

vocabulary = dict(nltk.Counter(vocabulary_list))
story = dict(nltk.Counter(story_list))
ask = dict(nltk.Counter(ask_list))
show = dict(nltk.Counter(show_list))
poll = dict(nltk.Counter(poll_list))

smooth_vocab = vocabulary.copy()
smooth_story = story.copy()
smooth_ask = ask.copy()
smooth_show = show.copy()
smooth_poll = poll.copy()

for i in smooth_vocab:
    if i not in smooth_story:
        smooth_story[i] = 0

    if i not in smooth_ask:
        smooth_ask[i] = 0

    if i not in smooth_show:
        smooth_show[i] = 0
    if i not in smooth_poll:
        smooth_poll[i] = 0



end_time = time.time()
print(end_time-start_time)
