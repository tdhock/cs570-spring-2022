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

import torch
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.weight_vec = torch.nn.Linear(ncol, 1)
    def forward(self, feature_mat):
        return self.weight_vec(feature_mat)

model = LinearModel()
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

optimizer.zero_grad()
batch_size = 10
feature_tensor = torch.from_numpy(scaled_features["subtrain"]).float()[:batch_size,:]
label_tensor = torch.from_numpy(set_labels["subtrain"])[:batch_size]
pred_vec = model(feature_tensor)
loss_value = loss_fun(pred_vec, label_tensor)
