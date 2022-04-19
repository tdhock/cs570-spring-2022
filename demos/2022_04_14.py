import torch
import torchtext
import numpy as np
import pandas as pd
torchtext.datasets.YelpReviewPolarity    
torchtext.datasets.AmazonReviewPolarity

# consider split='train' as full data set.
full = torchtext.datasets.IMDB(split='train')
items_list = list(full)
def get_size(item):
    label, text = item
    print("%s chars, %s words"%(len(text),len(text.split())))
get_size(items_list[1000])
get_size(items_list[20000])

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

[(k,word_dict[k]) for k in word_dict.keys()][:5]
max([count for word, count in word_dict.items()])
# TODO for increased prediction accuracy use a larger dictionary (but
# that is slower for the demo in class).
frequent_word_dict = {
    word:count 
    for word, count in word_dict.items() 
    if count > 99999}
word_to_ix = {
    word:ix 
    for ix, word in enumerate(frequent_word_dict)}

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

embed_dim = 4 # hyper-parameters!!! maybe try 32 or 64.
hidden_dim = 5
#not in dictionary of known words
other_ix = len(frequent_word_dict) 
vocab_size = other_ix+1 #extra is "other"
index_list = [word_to_ix.get(w,other_ix) for w in word_list]
index_tensor = torch.tensor(index_list, dtype=torch.long)
embed = torch.nn.Embedding(vocab_size, embed_dim)
embed_out = embed(index_tensor)
embed_out.shape
lstm = torch.nn.LSTM(embed_dim, hidden_dim) 
lstm_in = embed_out.view(len(word_list), 1, -1)
lstm_in.shape
embed_out.shape
lstm_out, (hidden, cell) = lstm(lstm_in)
lstm_out.shape
hidden.shape
cell.shape
linear =torch.nn.Linear(hidden_dim, 1)
pred_score = linear(hidden.reshape(1,hidden_dim))
label_int = 1 if label is "pos" else 0
label_tensor = torch.tensor([float(label_int)])
loss_fun = torch.nn.BCEWithLogitsLoss()
loss_value = loss_fun(pred_score.reshape(1), label_tensor)
## TODO connect to label.
loss_value.backward()
model = your_NN_class()
optimizer = torch.optim.SGD(model.parameters())
