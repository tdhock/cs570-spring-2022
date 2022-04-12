import torchtext
import numpy as np
import pandas as pd
torchtext.datasets.YelpReviewPolarity    
torchtext.datasets.AmazonReviewPolarity
train = torchtext.datasets.IMDB(split='train')
items_list = [item for item in train]
items_list[1000]
items_list[20000]

word_dict = {}
split_list = []
for label, text in items_list:
    word_list = text.split()
    for word in word_list:
        word_dict.setdefault(word, 0)
        word_dict[word] += 1
    item = label, word_list
    split_list.append(item)

max([count for word, count in word_dict.items()])
frequent_word_dict = {
    word:count for word, count in word_dict.items() if count > 99999}
len(frequent_word_dict)
word_to_ix = {word:ix for ix, word in enumerate(frequent_word_dict)}

for label, word_list in split_list:
    word_vec = np.zeros(len(word_to_ix))
    for word in word_list:
        if word in word_to_ix:
            ix = word_to_ix[word]
            word_vec[ix] += 1
