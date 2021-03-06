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
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.weight_vec = torch.nn.Linear(ncol, 1)
    def forward(self, feature_mat):
        return self.weight_vec(feature_mat)

# random initialization of weights.
model = LinearModel()
loss_fun = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)



pred_vec = model(set_features["subtrain"]).reshape(nrow)
loss_value = loss_fun(pred_vec, set_labels["subtrain"])
# loss_value is mean logistic loss.
loss_value
label_neg_pos = torch.where(set_labels["subtrain"] == 1, 1, -1)
torch.mean(torch.log(1+torch.exp(-label_neg_pos * pred_vec)))

# compute gradient
optimizer.zero_grad()
pred_vec = model(set_features["subtrain"]).reshape(nrow)
loss_value = loss_fun(pred_vec, set_labels["subtrain"])
loss_value.backward()
[p.shape for p in model.parameters()]
ncol
[p.grad for p in model.parameters()]
# exercise: verify that these gradients are the same as in Linear
# Models week.
optimizer.step()

class CSV(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, item):
        return self.features[item,:], self.labels[item]
    def __len__(self):
        return len(self.labels)

subtrain_dataset = CSV(set_features["subtrain"], set_labels["subtrain"])
len(subtrain_dataset)
nrow
subtrain_dataset[5] #calls getitem
batch_size = 20
subtrain_dataloader = torch.utils.data.DataLoader(
    subtrain_dataset, batch_size=batch_size, shuffle=True)

def compute_loss(features, labels):
    pred_vec = model(features)
    return loss_fun(
        pred_vec.reshape(len(pred_vec)),
        labels)

loss_df_list = []

# this is what happens in one epoch
for batch_features, batch_labels in subtrain_dataloader:
    optimizer.zero_grad()
    batch_loss = compute_loss(batch_features, batch_labels)
    batch_loss.backward()
    optimizer.step()

for set_name in set_features:
    feature_mat = set_features[set_name]
    label_vec = set_labels[set_name]
    set_loss = compute_loss(feature_mat, label_vec)
    loss_df_list.append(pd.DataFrame({
        "set_name":set_name,
        "loss":set_loss.detach().numpy(),
        }, index=[0]))

loss_df = pd.concat(loss_df_list)


loss_df_list = []
max_epochs=100
model = LinearModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
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
            "epoch":epoch,
            "set_name":set_name,
            "loss":set_loss.detach().numpy(),
            }, index=[0])
        print(set_df)
        loss_df_list.append(set_df)
loss_df = pd.concat(loss_df_list)

import plotnine as p9

gg = p9.ggplot()+\
    p9.geom_line(
        p9.aes(
            x="epoch",
            y="loss",
            color="set_name"
        ),
        data=loss_df)
show(gg)
