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
frequent_word_dict = {
    word:count 
    for word, count in word_dict.items() 
    if count > 999}
len(frequent_word_dict)
word_to_ix = {
    word:ix 
    for ix, word in enumerate(frequent_word_dict)}

embed_dim = 5 # hyper-parameters!!! maybe try 32 or 64.
hidden_dim = 10
#not in dictionary of known words
other_ix = len(frequent_word_dict) 
vocab_size = other_ix+1 #extra is "other"
class AttentionRNN(torch.nn.Module):
    def __init__(self):
        super(AttentionRNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.gru = torch.nn.GRU(embed_dim, hidden_dim) 
        self.process_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.U_to_V = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.last_linear = torch.nn.Linear(hidden_dim, 1)
    def forward(self, index_tensor):
        embed_out = self.embedding(index_tensor)
        n_items = len(index_tensor)
        gru_in = embed_out.view(n_items, 1, -1)
        gru_out, hidden = self.gru(gru_in)
        gru_out_processed = self.process_hidden(gru_out)
        hidden_act = torch.nn.Tanh()
        u_mat = hidden_act(gru_out_processed).reshape(n_items, hidden_dim)
        v_mat = self.U_to_V(u_mat)
        softmax = torch.nn.Softmax(dim=0)
        alpha_mat = softmax(v_mat).unsqueeze(-1)
        gru_out_norm = torch.mul(gru_out, alpha_mat.expand_as(gru_out))
        feature_vec = torch.sum(gru_out_norm, dim=0)
        pred_score = self.last_linear(feature_vec.reshape(1,hidden_dim))
        return pred_score, alpha_mat

model = AttentionRNN()
index_list = [word_to_ix.get(w,other_ix) for w in word_list]
index_tensor = torch.tensor(index_list, dtype=torch.long)
pred_score, norm_imp_weights = model(index_tensor)
importance_list = [
    (word,float(importance)) for importance, word in 
    zip(norm_imp_weights, word_list)]
importance_list.sort(key=lambda tup: tup[1])
importance_list[0]
importance_list[len(importance_list)-1]

label_int = 1 if label is "pos" else 0
label_tensor = torch.tensor([float(label_int)])
loss_fun = torch.nn.BCEWithLogitsLoss()
loss_value = loss_fun(pred_score.reshape(1), label_tensor)
loss_value.backward()

## subtrain/validation split.
np.random.seed(1)
import random
split_array = np.array(
    random.sample(split_list, 1000), dtype=object)
n_folds = 5
fold_vec = np.random.randint(
    low=0, high=n_folds, size=len(split_array))
validation_fold = 0
is_set_dict = {
    "validation":fold_vec == validation_fold,
    "subtrain":fold_vec != validation_fold,
}
set_list_dict = {}
for set_name, is_set in is_set_dict.items():
    set_array = split_array[is_set]
    set_list_dict[set_name] = []
    for label, word_list in set_array:
        index_list = [word_to_ix.get(w,other_ix) for w in word_list]
        index_tensor = torch.tensor(index_list, dtype=torch.long)
        label_int = 1 if label is "pos" else 0
        label_tensor = torch.tensor([float(label_int)])
        set_list_dict[set_name].append( (label_tensor, index_tensor) )
{set_name:len(set_list) for set_name,set_list in set_list_dict.items()}

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
max_epochs = 10
samples_per_print = 100
for epoch in range(max_epochs):
    n_steps = len(set_list_dict["subtrain"])
    for sample_i in range(n_steps):
        optimizer.zero_grad()
        label_tensor, index_tensor = set_list_dict["subtrain"][sample_i]
        pred_score, norm_imp_weights = model(index_tensor)
        loss_fun = torch.nn.BCEWithLogitsLoss()
        loss_value = loss_fun(pred_score.reshape(1), label_tensor)
        loss_value.backward()
        optimizer.step()
    epoch_loss_dict = {"epoch":epoch}
    with torch.no_grad():
        for set_name, set_list in set_list_dict.items():
            set_label_list = []
            set_pred_list = []
            for label_tensor, index_tensor in set_list:
                pred_tensor, norm_imp_weights = model(index_tensor)
                set_label_list.append(label_tensor)
                set_pred_list.append(pred_tensor)
            set_label_tensor = torch.cat(set_label_list)
            set_pred_tensor = torch.cat(
                set_pred_list).reshape(len(set_list))
            set_loss_tensor = loss_fun(
                set_pred_tensor, set_label_tensor)
            epoch_loss_dict[set_name] = float(set_loss_tensor)
    print("epoch=%(epoch)d subtrain=%(subtrain)f valid=%(validation)f"%epoch_loss_dict)

loss_text = """
epoch=0 subtrain=0.736914 valid=0.699236
epoch=1 subtrain=0.736004 valid=0.698525
epoch=2 subtrain=0.733049 valid=0.696248
epoch=3 subtrain=0.727531 valid=0.691829
epoch=4 subtrain=0.701838 valid=0.668578
epoch=5 subtrain=0.639475 valid=0.626510
epoch=6 subtrain=0.557031 valid=0.624553
epoch=7 subtrain=0.512921 valid=0.650581
epoch=8 subtrain=0.491017 valid=0.686842
epoch=9 subtrain=0.458467 valid=0.725732
"""
import re
loss_df_list = []
for line in loss_text.split("\n"):
    if line != "":
        my_dict = {}
        for text in line.split(" "):
            key, value = text.split("=")
            my_dict[key] = [float(value)]
        for set_name in "subtrain", "valid":
            set_dict = {
                "epoch":my_dict["epoch"],
                "set_name":[set_name],
                "loss":my_dict[set_name]}
            loss_df_list.append(pd.DataFrame(set_dict))
loss_df = pd.concat(loss_df_list)
import plotnine as p9
p9.ggplot()+\
    p9.geom_line(
        p9.aes(
            x="epoch",
            y="loss",
            color="set_name"
        ),
        data=loss_df)
