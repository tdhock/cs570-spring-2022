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
weight_vec = np.zeros(ncol)
pred_vec = np.matmul(scaled_features["subtrain"], weight_vec)
pred_vec.size
subtrain_labels = np.where(set_labels["subtrain"]==1, 1, -1)
grad_loss_pred = -subtrain_labels/(1+np.exp(subtrain_labels * pred_vec))
grad_loss_weight_mat = grad_loss_pred * scaled_features["subtrain"].T
grad_vec = grad_loss_weight_mat.sum(axis=1)
step_size = 0.1
weight_vec -= step_size * grad_vec
