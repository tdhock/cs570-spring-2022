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
test_fold = 0
is_set_dict = {
    "test":fold_vec == test_fold,
    "train":fold_vec != test_fold,
}

set_features = {}
set_labels = {}
for set_name, is_set in is_set_dict.items():
    set_features[set_name] = spam_features[is_set,:]
    set_labels[set_name] = spam_labels[is_set]
{set_name:array.shape for set_name, array in set_features.items()}

train_mean = set_features["train"].mean(axis=0)
train_sd = np.sqrt(set_features["train"].var(axis=0))

scaled_features = {
    set_name:(set_mat-train_mean)/train_sd
    for set_name, set_mat in set_features.items()
    }
{set_name:set_mat.mean(axis=0) for set_name, set_mat in scaled_features.items()}
{set_name:set_mat.var(axis=0) for set_name, set_mat in scaled_features.items()}

# 2. prediction for all test points instead of just one
n_test = set_labels["test"].size
test_predictions = np.empty(n_test)
for test_i in range(n_test):
    test_i_features = set_features["test"][test_i,:]
    set_labels["test"][test_i]
    diff_mat = set_features["train"] - test_i_features
    squared_diff_mat = diff_mat ** 2
    squared_diff_mat.shape
    squared_diff_mat.sum(axis=0) # sum over columns
    distance_vec = squared_diff_mat.sum(axis=1) # sum over rows
    distance_vec.shape
    sorted_indices = distance_vec.argsort()
    n_neighbors = 301
    if test_i == 5:
        pdb.set_trace()
    nearest_indices = sorted_indices[:n_neighbors]
    nearest_labels = set_labels["train"][nearest_indices]
    # 3. predicted probability and class
    pred_class = pd.Series(nearest_labels).value_counts().idxmax()
    pd.Series(nearest_labels).value_counts()/n_neighbors
    pred_prob1 = nearest_labels.mean()
    np.where(pred_prob1 < 0.5, 0, 1) #threshold prob to get predicted class.
    test_predictions[test_i] = pred_class
#compute accuracy in test set.
test_predictions == set_labels["test"]

class MyCV:
    def __init__(self, estimator, param_grid):
        self.estimator = estimator
        self.param_grid = param_grid
    def fit(self, X, y):
        pass

my_instance = MyCV("foo", 5)
my_instance.estimator
my_instance.param_grid
my_instance.some_random_attr = 6

s = """

this is

a python

string

"""
