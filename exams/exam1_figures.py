from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import plotnine as p9
# default in inches.
p9.options.figure_size=(4.5,2.5)
import numpy as np
np.set_printoptions(linewidth=55)
import pandas as pd
import os
# grid/contouring functions
def make_grid(mat, n_grid = 80):
    nrow, ncol = mat.shape
    assert ncol == 2
    mesh_args = mat.apply(
        lambda x: np.linspace(min(x),max(x), n_grid), axis=0)
    mesh_tup = np.meshgrid(*[mesh_args[x] for x in mesh_args])
    mesh_vectors = [v.flatten() for v in mesh_tup]
    return pd.DataFrame(dict(zip(mesh_args,mesh_vectors)))
# https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours
# https://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html#sphx-glr-auto-examples-edges-plot-contours-py
def contour_paths(three_cols, level):
    from skimage import measure
    uniq_df = three_cols.iloc[:,:2].apply(pd.unique)
    n_grid = uniq_df.shape[0]
    fun_mat = three_cols.iloc[:,2].to_numpy().reshape(
        [n_grid,n_grid]).transpose()
    contours = measure.find_contours(fun_mat, level)
    contour_df_list = []
    half_df = (uniq_df-uniq_df.diff()/2)[1:]
    half_df.index = [x-0.5 for x in half_df.index]
    lookup_df = pd.concat([uniq_df, half_df])
    for contour_i, contour_mat in enumerate(contours):
        one_contour_df = pd.DataFrame(contour_mat)
        one_contour_df.columns = [c+"_i" for c in uniq_df]
        one_contour_df["contour_i"] = contour_i
        for cname in lookup_df:
            iname = cname+"_i"
            contour_col = one_contour_df[iname]
            lookup_col = lookup_df[cname]
            index_df = lookup_col[contour_col].reset_index()
            one_contour_df[cname] = index_df[cname]
        contour_df_list.append(one_contour_df)
    return pd.concat(contour_df_list)

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

data_dict = {}

spam_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/spam.data",
    header=None, sep=" ")
nrow, ncol = spam_df.shape
label_col_num = ncol-1
col_num_vec = spam_df.columns.to_numpy()
label_vec = spam_df[label_col_num]
feature_mat = spam_df.iloc[:,col_num_vec != label_col_num]
feature_mat.columns = [f"word{col_num}" for col_num in feature_mat]
data_dict["spam"] = (feature_mat, label_vec)

zip_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/zip.test.gz",
    sep=" ", header=None)
label_col_num = 0
col_num_vec = zip_df.columns.to_numpy()
all_label_vec = zip_df[label_col_num]
is01 = all_label_vec.isin([0,1])
label_vec = all_label_vec[is01]
feature_mat = zip_df.loc[is01,col_num_vec != label_col_num]
data_dict["zip"] = (feature_mat, label_vec)

mixture_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/ESL.mixture.csv")
#mixture_df.query('party == "democratic" & height_in > 70')
label_col_name = "party"
col_name_vec = mixture_df.columns.to_numpy()
party_vec = mixture_df[label_col_name]
party_tuples = [
    ("democratic","blue",0),
    ("republican","red",1)
]
party_colors = {party:color for party,color,number in party_tuples}
party_number_dict = {party:number for party,color,number in party_tuples}
number_party_dict = {number:party for party,color,number in party_tuples}
def number_to_party_vec(v):
    return np.where(v==0, number_party_dict[0], number_party_dict[1])
label_vec = np.where(
    party_vec == "democratic",
    party_number_dict["democratic"],
    party_number_dict["republican"])
feature_mat = mixture_df.loc[:,col_name_vec != label_col_name]
data_dict["mixture"] = (feature_mat, label_vec)
pd.set_option("display.max_columns", 0)
mixture_df

mix_features, mix_labels = data_dict["mixture"]
grid_df = make_grid(mix_features)
grid_mat = grid_df.to_numpy()
neigh = KNeighborsClassifier(n_neighbors=1).fit(mix_features, mix_labels)
grid_df["prediction"] = neigh.predict(grid_mat)
grid_df["party"] = number_to_party_vec(grid_df.prediction)
gg = p9.ggplot()+\
    p9.theme_bw()+\
    p9.theme(subplots_adjust={'right': 0.7, 'bottom':0.2})+\
    p9.geom_point(
        p9.aes(
            x="height_in",
            y="weight_lb",
            color="party"
        ),
        size=0.1,
        data=grid_df)+\
    p9.geom_point(
        p9.aes(
            x="height_in",
            y="weight_lb",
            fill="party"
        ),
        color="black",
        size=2,
        data=mixture_df)+\
    p9.scale_color_manual(
        values=party_colors)+\
    p9.scale_fill_manual(
        values=party_colors)
show(gg)

neigh = KNeighborsClassifier(
    n_neighbors=100
).fit(mix_features, mix_labels)
grid_df["prediction"] = neigh.predict(grid_mat)
grid_df["party"] = number_to_party_vec(grid_df.prediction)
gg = p9.ggplot()+\
    p9.theme_bw()+\
    p9.theme(subplots_adjust={'right': 0.7, 'bottom':0.2})+\
    p9.geom_point(
        p9.aes(
            x="height_in",
            y="weight_lb",
            color="party"
        ),
        size=0.1,
        data=grid_df)+\
    p9.geom_point(
        p9.aes(
            x="height_in",
            y="weight_lb",
            fill="party"
        ),
        color="black",
        size=2,
        data=mixture_df)+\
    p9.scale_color_manual(
        values=party_colors)+\
    p9.scale_fill_manual(
        values=party_colors)
show(gg)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(mix_features, mix_labels)
grid_df["prediction"] = lr.predict(grid_mat)
grid_df["party"] = number_to_party_vec(grid_df.prediction)
gg = p9.ggplot()+\
    p9.theme_bw()+\
    p9.theme(subplots_adjust={'right': 0.7, 'bottom':0.2})+\
    p9.geom_point(
        p9.aes(
            x="height_in",
            y="weight_lb",
            color="party"
        ),
        size=0.1,
        data=grid_df)+\
    p9.geom_point(
        p9.aes(
            x="height_in",
            y="weight_lb",
            fill="party"
        ),
        color="black",
        size=2,
        data=mixture_df)+\
    p9.scale_color_manual(
        values=party_colors)+\
    p9.scale_fill_manual(
        values=party_colors)
show(gg)

feature_mat = mix_features.to_numpy()
mean_vec = feature_mat.mean(axis=0)
sd_vec = np.sqrt(feature_mat.var(axis=0))
scaled_mat = (feature_mat - mean_vec)/sd_vec
n_train = mix_labels.size
learn_mat = np.column_stack((np.repeat(1, n_train), scaled_mat))
label_vec = np.where(mix_labels==1, 1, -1)
max_iterations = 100
weight_vec = np.repeat(0.0, feature_mat.shape[1]+1)
step_size = 10
sk_pred = lr.decision_function(feature_mat)
def log_deriv(pred, label):
    return -label / (1+np.exp(pred*label))
deriv_dict = {
    "logistic":log_deriv,
    "square":lambda f, y: 2*(f-y),
}
fun_dict = {
    "logistic":lambda f, y: np.log(1+np.exp(-y*f)),
    "zero-one":lambda f, y: np.where(f>0, 1, -1)!=y,
}
sk_loss = fun_dict["logistic"](sk_pred, label_vec).mean()
loss_df_list = []
for iteration in range(max_iterations):
    pred_vec = np.matmul(learn_mat, weight_vec)
    deriv_loss_pred = deriv_dict["logistic"](pred_vec, label_vec)
    deriv_mat = deriv_loss_pred.reshape(n_train,1) * learn_mat
    grad_vec = deriv_mat.mean(axis=0)
    weight_vec -= step_size * grad_vec
    loss_vec = fun_dict["logistic"](pred_vec, label_vec)
    print('iteration=%d loss=%f gsum=%f'%(iteration,loss_vec.mean(),np.abs(grad_vec).sum()))
    orig_weights = weight_vec[1:]/sd_vec
    orig_intercept = weight_vec[0]-(mean_vec/sd_vec*weight_vec[1:]).sum()
    loss_df_list.append(pd.DataFrame({
        "iteration":iteration,
        "slope":-orig_weights[0]/orig_weights[1],
        "intercept":-orig_intercept/orig_weights[1],
        "loss":loss_vec.mean(),
    }, index=[0]))
    #np.matmul(feature_mat, orig_weights)+orig_intercept
    #np.matmul(learn_mat, weight_vec)
loss_df = pd.concat(loss_df_list)


mixture_noise = mixture_df.copy()
nrow, col = mixture_noise.shape
feature_names = ["height_in","weight_lb"]
for n_noise in range(20):
    np.random.seed(n_noise)
    fname = f"noise{n_noise}"
    feature_names.append(fname)
    mixture_noise[fname] = np.random.randn(nrow)*100
pd.set_option("display.max_columns", 4)
mixture_noise

mixture_noise["set"] = np.resize(["subtrain","validation"], n_train)
is_subtrain = mixture_noise.set == "subtrain"
input_features = mixture_noise.loc[:,feature_names].to_numpy()
subtrain_features = input_features[is_subtrain,:]
subtrain_labels = label_vec[is_subtrain]
n_subtrain=subtrain_labels.size
mean_vec = subtrain_features.mean(axis=0)
sd_vec = np.sqrt(subtrain_features.var(axis=0))
scaled_mat = (subtrain_features - mean_vec)/sd_vec
learn_mat = np.column_stack((np.repeat(1, n_subtrain), scaled_mat))
max_iterations = 100
step_size = 50
weight_vec = np.repeat(0.0, scaled_mat.shape[1]+1)
loss_df_list = []
for iteration in range(max_iterations):
    subtrain_pred = np.matmul(learn_mat, weight_vec)
    deriv_loss_pred = deriv_dict["logistic"](subtrain_pred, subtrain_labels)
    deriv_mat = deriv_loss_pred.reshape(n_subtrain,1) * learn_mat
    grad_vec = deriv_mat.mean(axis=0)
    weight_vec -= step_size * grad_vec
    orig_weights = weight_vec[1:]/sd_vec
    orig_intercept = weight_vec[0]-(mean_vec/sd_vec*weight_vec[1:]).sum()
    input_pred = np.matmul(input_features, orig_weights) + orig_intercept
    loss_vec = fun_dict["logistic"](input_pred, label_vec)
    iteration_set_loss = pd.DataFrame({
        "mean_loss":loss_vec,
        "set":mixture_noise.set,
    }).groupby("set").mean().reset_index()
    iteration_set_loss["iteration"] = iteration
    print(iteration_set_loss)
    loss_df_list.append(iteration_set_loss)
loss_df = pd.concat(loss_df_list)

valid = loss_df.query("set=='validation'")
min_valid = valid.iloc[[valid.mean_loss.argmin()]]
min_valid["label"] = "min at %d iterations"%min_valid.iteration
gg=p9.ggplot()+\
    p9.theme(subplots_adjust={'right': 0.7, "bottom":0.2})+\
    p9.geom_line(
        p9.aes(
            x="iteration",
            y="mean_loss",
            color="set"
            ),
        data=loss_df)
show(gg)
gg.save("exam1_subtrain_validation_loss_too_big.png")
