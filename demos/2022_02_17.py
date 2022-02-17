import pandas as pd
import numpy as np

spam_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/spam.data",
    header=None,
    sep=" ")

spam_features = spam_df.iloc[:,:-1].to_numpy()
spam_labels = spam_df.iloc[:,-1].to_numpy()
# 1. feature scaling
spam_features.mean()
spam_features.mean(axis=0)
spam_features.var(axis=0) # var = sd^2
np.sqrt(spam_features.var(axis=0))

np.random.seed(1)
n_folds = 5
fold_vec = np.random.randint(low=0, high=n_folds, size=spam_labels.size)
validation_fold = 0
is_set_dict = {
    "validation":fold_vec == validation_fold,
    "subtrain":fold_vec != validation_fold,
}

set_features = {}
set_labels = {}
for set_name, is_set in is_set_dict.items():
    set_features[set_name] = spam_features[is_set,:]
    set_labels[set_name] = spam_labels[is_set]
{set_name:array.shape for set_name, array in set_features.items()}

subtrain_mean = set_features["subtrain"].mean(axis=0)
subtrain_sd = np.sqrt(set_features["subtrain"].var(axis=0))

scaled_features = {
    set_name:(set_mat-subtrain_mean)/subtrain_sd
    for set_name, set_mat in set_features.items()
    }
{set_name:set_mat.mean(axis=0) for set_name, set_mat in scaled_features.items()}
{set_name:set_mat.var(axis=0) for set_name, set_mat in scaled_features.items()}

nrow, ncol = scaled_features["subtrain"].shape

units_per_layer = [ncol, 100, 10, 1]
# initialization
weight_mat_list = []
for layer_i in range(1, len(units_per_layer)):
    print(layer_i)
    w_size = units_per_layer[layer_i], units_per_layer[layer_i-1]
    print(w_size)
    weight_mat_list.append(np.random.normal(size=w_size))

# forward propagation.
h_list = [ set_features["subtrain"] ]
a_list = []
for layer_i in range(1, len(units_per_layer)):
    prev_layer_h = h_list[layer_i-1]
    layer_weight_mat = weight_mat_list[layer_i-1].T
    layer_a = np.matmul(prev_layer_h, layer_weight_mat)
    a_list.append(layer_a)
    if layer_i == len(units_per_layer)-1:
        layer_h = layer_a #identity activation for last layer
    else:
        layer_h = np.where(layer_a > 0, layer_a, 0) 
    h_list.append(layer_h)
[hmat.shape for hmat in h_list]

# back propagation.
grad_w_list = []
subtrain_labels = set_labels["subtrain"].reshape(layer_h.size, 1)
grad_loss_last_h = -subtrain_labels/(
    1+np.exp(subtrain_labels * layer_h))
grad_loss_last_h.shape
grad_h = grad_loss_last_h
for layer_i in range(len(units_per_layer)-1, 0, -1):
    # grad_a is nrow_batch x units_this_layer
    layer_a = a_list[layer_i-1]
    grad_act = np.where( #gradient of relu activation
        layer_a < 0,
        1 if layer_i == len(units_per_layer)-1 else 0,
        1)
    grad_a = grad_act * grad_h
    grad_w = TODO #rule 1.
    grad_h = TODO #rule 3.
    grad_w_list.append(grad_w) # at the beginning would be better
    print(layer_i)
    
