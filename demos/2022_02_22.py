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

grad_loss_last_h = -subtrain_labels/(
    1+np.exp(subtrain_labels * layer_h))

class InitialNode:
    def __init__(self, value):
        self.value = value

class Operation:
    def __init__(self, *args):
        for input_name, node in zip(self.input_names, args):
            setattr(self, input_name, node)
        self.value = self.forward()
    def backward(self):
        tuple_of_gradients = self.gradient()
        for input_name, grad in zip(self.input_names, tuple_of_gradients):
            node = getattr(self, input_name)
            node.grad = grad
                 
class mm(Operation):
    input_names = ["features", "weights"]
    def forward(self):
        return np.matmul(self.features.value, self.weights.value)
    def gradient(self):
        return (
            np.matmul(self.grad, self.weights.value.T),
            np.matmul(self.features.value.T, self.grad)
        )

class logistic_loss(Operation):
    input_names = ["predictions", "labels"]
    
weight_node = InitialNode(np.random.normal(size=ncol))
label_node = InitialNode(set_labels["subtrain"])
feature_node = InitialNode(scaled_features["subtrain"])
pred_node = mm(feature_node, weight_node)
loss_node = logistic_loss(pred_node, label_node)
loss_node.backward()
weight_node.grad should be assigned now!
