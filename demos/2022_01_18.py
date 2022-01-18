import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

zip_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/zip.test.gz",
    header=None,
    sep=" ")

is01 = zip_df[0].isin([0,1])
zip01_df = zip_df.loc[is01,:]

data_dict = {
    "zip":(zip01_df.loc[:,1:].to_numpy(), zip01_df[0]),
    #"spam":TODO
}

for data_set, (input_mat, output_vec) in data_dict.items():
    print(data_set)
    kf = KFold(n_splits=3, shuffle=True, random_state=1)
    for fold_id, (train_index, test_index) in enumerate(kf.split(input_mat)):
        print({fold_id:test_index})
        parameters = {'n_neighbors':[x for x in range(1, 21)]}
        clf = GridSearchCV(KNeighborsClassifier(), parameters)
        train_input_mat = TODO input_mat [ train_index ]
        clf.fit(train_input_mat, train_output_vec)
        
