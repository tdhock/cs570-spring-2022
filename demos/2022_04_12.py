import torch
import torchtext
import numpy as np
import pandas as pd
torchtext.datasets.YelpReviewPolarity    
torchtext.datasets.AmazonReviewPolarity

# consider split='train' as full data set.
full = torchtext.datasets.IMDB(split='train')
items_list = [item for item in full]
items_list[1000]
items_list[20000]

word_dict = {}
split_list = []
for label, text in items_list:
    # TODO maybe remove HTML tags + punctuation from text before
    # splitting into words, to reduce noise.
    word_list = text.split()
    for word in word_list:
        word_dict.setdefault(word, 0)
        word_dict[word] += 1
    item = label, word_list
    split_list.append(item)

[k for k in word_dict.keys()][:5]
max([count for word, count in word_dict.items()])
# TODO for increased prediction accuracy use a larger dictionary (but
# that is slower for the demo in class).
frequent_word_dict = {
    word:count for word, count in word_dict.items() if count > 99999}
len(frequent_word_dict)
word_to_ix = {word:ix for ix, word in enumerate(frequent_word_dict)}

feature_list = []
label_list = []
for label, word_list in split_list:
    word_vec = np.zeros(len(word_to_ix))
    for word in word_list:
        if word in word_to_ix:
            ix = word_to_ix[word]
            word_vec[ix] += 1
    feature_list.append(word_vec)
    label_list.append(label)
data_dict = {"IMDB":(np.array(feature_list), np.array(label_list))}
[array.shape for array in data_dict["IMDB"] ]

# code from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
lstm = torch.nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
## TODO use our inputs from above instead of random inputs here.
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
[input.shape for input in inputs]
# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    lstm_input = i.view(1, 1, -1)
    out, hidden = lstm(lstm_input, hidden)

