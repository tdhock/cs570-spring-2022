import pandas as pd
import numpy as np

zip_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/zip.test.gz",
    header=None,
    sep=" ")

is01 = zip_df[0].isin([0,1])
zip01_df = zip_df.loc[is01,:]

zip_features = zip01_df.loc[:,1:].to_numpy()
zip_labels = zip01_df[0].to_numpy()

dir(np.random)
help(np.random.randint)
help(np.random.set_state)
np.random.seed(1)
n_folds = 5
fold_vec = np.random.randint(low=0, high=n_folds, size=zip_labels.size)
test_fold = 0
is_set_dict = {
    "test":fold_vec == test_fold,
    "train":fold_vec != test_fold,
}

set_features = {
    set_name:zip_features[is_set,:]
    for set_name, is_set in is_set_dict.items()
}
{set_name:array.shape for set_name, array in set_features.items()}
set_labels = {
    set_name:zip_labels[is_set]
    for set_name, is_set in is_set_dict.items()
}

test_i = 0
test_i_features = set_features["test"][test_i,:]
diff_mat = set_features["train"] - test_i_features
squared_diff_mat = diff_mat ** 2
dir(diff_mat)
help(diff_mat.sum)
squared_diff_mat.sum(axis=0) # sum over columns
distance_vec = squared_diff_mat.sum(axis=1) # sum over rows
sorted_indices = distance_vec.argsort()
n_neighbors = 301
nearest_indices = sorted_indices[:n_neighbors]
nearest_labels = set_labels["train"][nearest_indices]
