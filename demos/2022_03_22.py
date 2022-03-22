import pandas as pd
import numpy as np

# work-around for rendering plots under windows, which hangs within
# emacs python shell: instead write a PNG file and view in browser.
import webbrowser
on_windows = os.name == "nt"
in_render = r.in_render if 'r' in dir() else False
using_agg = on_windows and not in_render
if using_agg:
    import matplotlib
    matplotlib.use("agg")
def show(g):
    if not using_agg:
        return g
    g.save("tmp.png")
    webbrowser.open('tmp.png')


spam_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/spam.data",
    header=None,
    sep=" ")

spam_features = spam_df.iloc[:,:-1].to_numpy()
spam_labels = spam_df.iloc[:,-1].to_numpy()
# 1. feature scaling
spam_mean = spam_features.mean(axis=0)
spam_sd = np.sqrt(spam_features.var(axis=0))
scaled_features = (spam_features-spam_mean)/spam_sd
scaled_features.mean(axis=0)
scaled_features.var(axis=0)

np.random.seed(1)
n_folds = 5
fold_vec = np.random.randint(low=0, high=n_folds, size=spam_labels.size)
validation_fold = 0
is_set_dict = {
    "validation":fold_vec == validation_fold,
    "subtrain":fold_vec != validation_fold,
}

import torch
set_features = {}
set_labels = {}
for set_name, is_set in is_set_dict.items():
    set_features[set_name] = torch.from_numpy(
        scaled_features[is_set,:]).float()
    set_labels[set_name] = torch.from_numpy(
        spam_labels[is_set]).float()
{set_name:array.shape for set_name, array in set_features.items()}

nrow, ncol = set_features["subtrain"].shape
class NNet(torch.nn.Module):
    def __init__(self, n_hidden_units):
        super(NNet, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(ncol, n_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_units, 1))
    def forward(self, feature_mat):
        return self.seq(feature_mat)

# random initialization of weights.
model = NNet(1)
loss_fun = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

class CSV(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, item):
        return self.features[item,:], self.labels[item]
    def __len__(self):
        return len(self.labels)

subtrain_dataset = CSV(set_features["subtrain"], set_labels["subtrain"])
batch_size = 20
subtrain_dataloader = torch.utils.data.DataLoader(
    subtrain_dataset, batch_size=batch_size, shuffle=True)

def compute_loss(features, labels):
    pred_vec = model(features)
    return loss_fun(
        pred_vec.reshape(len(pred_vec)),
        labels)

loss_df_dict = {}
max_epochs=20

for n_hidden_units in [1,5,10,50,80,100,200,500,1000,5000]:
    if (n_hidden_units,"validation") not in loss_df_dict:
        model = NNet(n_hidden_units)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        for epoch in range(max_epochs):
            # first update weights.
            for batch_features, batch_labels in subtrain_dataloader:
                optimizer.zero_grad()
                batch_loss = compute_loss(batch_features, batch_labels)
                batch_loss.backward()
                optimizer.step()
        # then compute subtrain/validation loss.
        for set_name in set_features:
            feature_mat = set_features[set_name]
            label_vec = set_labels[set_name]
            set_loss = compute_loss(feature_mat, label_vec)
            set_df = pd.DataFrame({
                "n_hidden_units":n_hidden_units,
                "set_name":set_name,
                "loss":set_loss.detach().numpy(),
                }, index=[0])
            print(set_df)
            loss_df_dict[(n_hidden_units,set_name)] = set_df
loss_df = pd.concat(loss_df_dict.values())

import plotnine as p9

gg = p9.ggplot()+\
    p9.scale_x_log10()+\
    p9.geom_line(
        p9.aes(
            x="n_hidden_units",
            y="loss",
            color="set_name"
        ),
        data=loss_df)
show(gg)
