import pandas as pd
import nltk
from collections import OrderedDict
from collections import Counter
import time
import numpy as np

# -------------------------------------------------------
# Assignment 2
# Written by Charles Abou Haidar, 40024373
# For COMP 472 – Summer 2020
# --------------------------------------------------------

if not nltk: # download nltk if not available
    nltk.download()

SMOOTHING_VALUE = 0.5

# FUNCTIONS THAT WILL BE USED ACROSS THE ASSIGNMENT...
"""
function that takes in a list and returns all elements of that list lowered
"""


def lower_list_elements(list_to_lower):
    lowered_list = [element.lower() for element in list_to_lower]
    return lowered_list


"""
function that takes in a list and returns all elements of that list split into strings delimited by whitespace
"""


def tokenize_and_regexp(list1):
    tokenized_list = []

    for sentence in list1:
        tokenized_list.append(nltk.regexp_tokenize(sentence, r"\w+"))

    return tokenized_list


"""
apply smoothing for vocab
"""


def apply_smoothing(vocab):
    for j in vocab:
        vocab[j] += SMOOTHING_VALUE

    return vocab


"""
flatten a list
e.g
[[1, 2, 3], [4, 5, 6]] becomes [1, 2, 3, 4, 5, 6]
"""


def flatten_list(list1):
    flattened_list = [item for sublist in list1 for item in sublist]

    return flattened_list


"""
smooth words function
"""


def smooth_words(list1):
    total = 0

    for i in list1:
        total += list1[i]

    return total


"""
write classifier
"""


def write_to_file1(file_name, story1, ask1, show1, poll1, total1):
    file_name_open = open(file_name, "w+", encoding="UTF-8")
    line_counter = 0
    for i in range(len(testing_set)):
        title = testing_set['Title'].iloc[i].lower()  # get title and lower them
        post_type = testing_set['Post Type'].iloc[i]  # get post type
        words_list = title.split()  # split title into words

        story_freq = len(story1) / total1  # calculate story requency
        ask_freq = len(ask1) / total1  # calculate ask frequency
        show_freq = len(show1) / total1  # calculate show frequency
        poll_freq = len(poll1) / total1  # calculate poll frequency

        # story frequency
        if story_freq == 0:  # check if no occurence of story, give it - infinity
            story_score = float("-inf")
        else:  # use log10 as indicated by teacher
            story_score = np.log10(story_freq)

        # ask frequency
        if ask_freq == 0:  # check if no occurences of ask, give it - infinity
            ask_score = float("-inf")
        else:  # use log10 as indicated by teacher
            ask_score = np.log10(ask_freq)

        # show frequency
        if show_freq == 0:  # check if no occurences of show, give it - infinity
            show_score = float("-inf")
        else:  # use log10 as indicated by teacher
            show_score = np.log10(show_freq)

        # poll frequency
        if poll_freq == 0:  # check if no occurences of poll, give it - infinity
            poll_score = 0
        else:  # use log10 as indicated by teacher
            poll_score = np.log10(poll_freq)

        for word in words_list:

            if word in story_probabilities:
                story_score += np.log10(story_probabilities[word])

            if word in ask_probabilities:
                ask_score += np.log10(ask_probabilities[word])

            if word in show_probabilities:
                show_score += np.log10(show_probabilities[word])

            if word in poll_probabilities:
                poll_score += np.log10(poll_probabilities[word])

        classification = {"story": story_score, "ask_hn": ask_score, "show_hn": show_score, "poll": poll_score}
        file_name_open.write(str(line_counter) + "\t\t" +
                             title + "\t\t" +
                             max(classification, key=classification.get) + "\t\t" +
                             str(classification["story"]) + "\t\t" +
                             str(classification["ask_hn"]) + "\t\t" +
                             str(classification["show_hn"]) + "\t\t" +
                             str(classification["poll"]) + "\t\t" +
                             post_type + "\t\t")

        if max(classification, key=classification.get) == post_type:
            file_name_open.write("right" + "\n")
        else:
            file_name_open.write("wrong" + "\n")

        line_counter += 1

    file_name_open.close()



# main (assignment officially starts here) #

# TASK 1 START

start_time = time.time()  # start time
reader = pd.read_csv("hns_2018_2019.csv")  # open csv file
data_frame = pd.DataFrame(reader)  # convert csv file to data frame

# get all info from dataframe where the year is 2018 for training set
training_set = data_frame.loc[data_frame['Created At'].str.contains("2018")]

# get all the info from dataframe where the year is 2019 for testings et
testing_set = data_frame.loc[data_frame['Created At'].str.contains("2019")]

# extracting all the title text from all post types and converting them to a list

# get all the story post types in 2018
story = training_set['Title'].loc[training_set['Post Type'] == 'story'].values.tolist()

# get all the ask_hn post types in 2018
ask = training_set['Title'].loc[training_set['Post Type'] == 'ask_hn'].values.tolist()

# get all the show_hn post types in 2018
show = training_set['Title'].loc[training_set['Post Type'] == 'show_hn'].values.tolist()

# get all the poll post types in 2018
poll = training_set['Title'].loc[training_set['Post Type'] == 'poll'].values.tolist()

# converting all post type Titles into lower case
story = lower_list_elements(story)
ask = lower_list_elements(ask)
show = lower_list_elements(show)
poll = lower_list_elements(poll)

# splitting all elements in list of Title post types into separate words delimited by whitespace
story_list = tokenize_and_regexp(story)
ask_list = tokenize_and_regexp(ask)
show_list = tokenize_and_regexp(show)
poll_list = tokenize_and_regexp(poll)

# flatten each list
story_list = flatten_list(story_list)
ask_list = flatten_list(ask_list)
show_list = flatten_list(show_list)
poll_list = flatten_list(poll_list)

vocabulary = story_list + ask_list + show_list + poll_list

# create dictionaries for each post type from the list that has the splitted words
vocab_dict = dict(Counter(vocabulary))
story_dict = dict(Counter(story_list))
ask_dict = dict(Counter(ask_list))
show_dict = dict(Counter(show_list))
poll_dict = dict(Counter(poll_list))

# create smoothed vocabulary for
smooth_vocab = vocab_dict.copy()
smooth_story = story_dict.copy()
smooth_ask = ask_dict.copy()
smooth_show = show_dict.copy()
smooth_poll = poll_dict.copy()

# add missing words for smoothing
for i in smooth_vocab:
    if i not in smooth_story:
        smooth_story[i] = 0

    if i not in smooth_ask:
        smooth_ask[i] = 0

    if i not in smooth_show:
        smooth_show[i] = 0

    if i not in smooth_poll:
        smooth_poll[i] = 0

# apply smoothing to each word in vocab/post types
smooth_vocab = apply_smoothing(smooth_vocab)
smooth_story = apply_smoothing(smooth_story)
smooth_ask = apply_smoothing(smooth_ask)
smooth_show = apply_smoothing(smooth_show)
smooth_poll = apply_smoothing(smooth_poll)

# create smooth dictionary for each post type and vocab
smooth_vocab = dict(OrderedDict(sorted(smooth_vocab.items())))
smooth_story = dict(OrderedDict(sorted(smooth_story.items())))
smooth_ask = dict(OrderedDict(sorted(smooth_ask.items())))
smooth_show = dict(OrderedDict(sorted(smooth_show.items())))
smooth_poll = dict(OrderedDict(sorted(smooth_poll.items())))

# remove words that have numbers and special characters such as a period, dash or apostrophe
remove_words = []
remove_words_doc = open("remove_words.txt", "w+", encoding="UTF-8")
line_counter = 1
for i in list(smooth_vocab):
    if (not i.isalpha()) and ("." not in i) and ("-" not in i) and ("'" not in i):
        smooth_vocab.pop(i)
        smooth_story.pop(i)
        smooth_ask.pop(i)
        smooth_show.pop(i)
        smooth_poll.pop(i)
        remove_words.append(i)
        remove_words_doc.write(str(line_counter) + "\t" + str(i) + "\n")
        line_counter += 1

remove_words_doc.close()

smooth_vocab_count = smooth_words(smooth_vocab)
smooth_story_count = smooth_words(smooth_story)
smooth_ask_count = smooth_words(smooth_ask)
smooth_show_count = smooth_words(smooth_show)
smooth_poll_count = smooth_words(smooth_poll)

# create dictionary to hold probabilities of each post type
story_probabilities = dict()
ask_probabilities = dict()
show_probabilities = dict()
poll_probabilities = dict()

# open txt files to write in them
model_2018_file = open("model-2018.txt", "w+", encoding="UTF-8")
vocabulary_file = open("vocabulary.txt", "w+", encoding="UTF-8")
line_counter = 0

for i in smooth_vocab:
    story_probabilities[str(i)] = smooth_story[i] / smooth_story_count
    ask_probabilities[str(i)] = smooth_ask[i] / smooth_ask_count
    show_probabilities[str(i)] = smooth_show[i] / smooth_show_count
    poll_probabilities[str(i)] = smooth_poll[i] / smooth_poll_count
    model_2018_file.write(str(line_counter) + "\t\t" + str(i) + "\t\t" +
                          str(smooth_story[i]) + "\t\t" + str(story_probabilities[i]) + "\t" +
                          str(smooth_ask[i]) + "\t\t" + str(ask_probabilities[i]) + "\t" +
                          str(smooth_show[i]) + "\t\t" + str(show_probabilities[i]) + "\t" +
                          str(smooth_poll[i]) + "\t\t" + str(poll_probabilities[i]) + "\n")

    vocabulary_file.write(str(line_counter) + "\t" + str(i) + "\n")
    line_counter += 1

model_2018_file.close()
vocabulary_file.close()

# TASK 1 END

# TASK 2 START
total_post_type_len = len(story) + len(ask) + len(show) + len(poll)
base_line_result_file = "baseline-result.txt"
write_to_file1(file_name=base_line_result_file, story1=story, ask1=ask, show1=show, poll1=poll,
               total1=total_post_type_len)
# END TASK 2

# START TASK 3
# START EXPERIMENT 1

stop_word_file = open("stopwords.txt")
stop_word_vocab = [word for line in stop_word_file for word in line.split()]

stop_word_smoothed_vocab = smooth_vocab.copy()
stop_word_smoothed_story = smooth_story.copy()
stop_word_smoothed_ask = smooth_ask.copy()
stop_word_smoothed_show = smooth_show.copy()
stop_word_smoothed_poll = smooth_poll.copy()

for word in stop_word_vocab:
    if word in list(stop_word_smoothed_vocab):
        stop_word_smoothed_vocab.pop(word)
        stop_word_smoothed_story.pop(word)
        stop_word_smoothed_ask.pop(word)
        stop_word_smoothed_show.pop(word)
        stop_word_smoothed_poll.pop(word)

# smooth word list for each post type
smooth_vocab_count = smooth_words(stop_word_smoothed_vocab)
smooth_story_count = smooth_words(stop_word_smoothed_story)
smooth_ask_count = smooth_words(stop_word_smoothed_ask)
smooth_show_count = smooth_words(stop_word_smoothed_show)
smooth_poll_count = smooth_words(stop_word_smoothed_poll)

# dictionaries for each stop word post ty pe
stop_word_prob_story = dict()
stop_word_prob_ask = dict()
stop_word_prob_show = dict()
stop_word_prob_poll = dict()

stop_word_model_file = open("stopword-model.txt", "w+", encoding="UTF-8")
line_counter = 0

for word in stop_word_smoothed_vocab:
    stop_word_prob_story[str(word)] = stop_word_smoothed_story[word] / smooth_story_count
    stop_word_prob_ask[str(word)] = stop_word_smoothed_ask[word] / smooth_ask_count
    stop_word_prob_show[str(word)] = stop_word_smoothed_show[word] / smooth_show_count
    stop_word_prob_poll[str(word)] = stop_word_smoothed_poll[word] / smooth_poll_count
    stop_word_model_file.write(str(line_counter) + "\t" +
                               str(word) + "\t" +
                               str(stop_word_smoothed_story[word]) + "\t" +
                               str(stop_word_prob_story[str(word)]) + "\t" +
                               str(stop_word_smoothed_ask[word]) + "\t" +
                               str(stop_word_prob_ask[str(word)]) + "\t" +
                               str(stop_word_smoothed_show[word]) + "\t" +
                               str(stop_word_prob_show[str(word)]) + "\t" +
                               str(stop_word_smoothed_poll[word]) + "\t" +
                               str(stop_word_prob_poll[str(word)]) + "\n")
    line_counter += 1

stop_word_model_file.close()

stop_word_result_file = "stopword-result.txt"
# write stop-result.txt file
write_to_file1(file_name=stop_word_result_file, story1=story, ask1=ask, show1=show, poll1=poll, total1=total_post_type_len)

# END EXPERIMENT 1

# START EXPERIMENT 2
word_length_smoothed_vocab = smooth_vocab.copy()
word_length_smoothed_story = smooth_story.copy()
word_length_smoothed_ask = smooth_ask.copy()
word_length_smoothed_show = smooth_show.copy()
word_length_smoothed_poll = smooth_poll.copy()

# remove words that are less than 3 characters long or bigger than 8
for word in list(word_length_smoothed_vocab):
    if len(word) < 3 or len(word) > 8:
        word_length_smoothed_vocab.pop(word)
        word_length_smoothed_story.pop(word)
        word_length_smoothed_ask.pop(word)
        word_length_smoothed_show.pop(word)
        word_length_smoothed_poll.pop(word)

# smooth word list for each post type
smooth_vocab_count = smooth_words(word_length_smoothed_vocab)
smooth_story_count = smooth_words(word_length_smoothed_story)
smooth_ask_count = smooth_words(word_length_smoothed_ask)
smooth_show_count = smooth_words(word_length_smoothed_show)
smooth_poll_count = smooth_words(word_length_smoothed_poll)

# probabilities dictionaries for each word length post ype
word_length_prob_story = dict()
word_length_prob_ask = dict()
word_length_prob_show = dict()
word_length_prob_poll = dict()

word_length_model_file = open("wordlength-model.txt", "w+", encoding="UTF-8")
line_counter = 0

# write into wordlength-model.txt
for word in word_length_smoothed_vocab:
    word_length_prob_story[str(word)] = word_length_smoothed_story[word] / smooth_story_count
    word_length_prob_ask[str(word)] = word_length_smoothed_ask[word] / smooth_ask_count
    word_length_prob_show[str(word)] = word_length_smoothed_show[word] / smooth_show_count
    word_length_prob_poll[str(word)] = word_length_smoothed_poll[word] / smooth_poll_count
    word_length_model_file.write(str(line_counter) + "\t\t" +
                                 str(word) + "\t\t" +
                                 str(word_length_smoothed_story[word]) + "\t\t" +
                                 str(word_length_prob_story[str(word)]) + "\t\t" +
                                 str(word_length_smoothed_ask[word]) + "\t\t" +
                                 str(word_length_prob_ask[str(word)]) + "\t\t" +
                                 str(word_length_smoothed_show[word]) + "\t\t" +
                                 str(word_length_prob_show[str(word)]) + "\t\t" +
                                 str(word_length_smoothed_poll[word]) + "\t\t" +
                                 str(word_length_prob_poll[str(word)]) + "\n")
    line_counter += 1

word_length_model_file.close()

word_length_result_file = "wordlength-result.txt"
write_to_file1(file_name=word_length_result_file, story1=story, ask1=ask, show1=show, poll1=poll, total1=total_post_type_len)

# END EXPERIMENT 2

# START EXPERIMENT 3
"""
TODO:
Frequency = 1
Frequency <= 5
Frequency <= 10
Frequency <= 15
Frequency <= 20
Remove top 5, 10, 15, 20 and 25% most frequent words from vocab in baseline
plot performance (accuracy, precision, recall, f-score)
"""
# END EXPERIMENT 3


end_time = time.time()
print(end_time - start_time)

# EOF
