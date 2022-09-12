# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import SelectFromModel, RFECV, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import model_selection
import pathlib
import scipy
import random
import warnings
from tqdm import tqdm
import dask
from dask import delayed
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from distributed import Client

warnings.filterwarnings("ignore")
sns.set()
from cellesce import Cellesce

plt.ion()

VARIABLES = ["Conc /uM", "Date", "Drug"]
SAVE_FIG = True
SAVE_CSV = True

# data_folder = "analysed/210720 - ISO49+34 - projection_XY/unet_2022/project_XY_all"
# data_folder = "analysed/210720 - ISO49+34 - projection_XY/unet_2022/project_XY_objects"

# pd.read_csv("analysed/_2019_cellesce_unet_splineparameters_aligned/raw/projection_XY/Secondary.csv")

kwargs = {
    "data_folder": "analysed/210720 - ISO49+34 - projection_XY/unet_2022/project_XY_all",
    "nuclei_path": "object_filteredNuclei.csv",
}
kwargs = {
    "data_folder": "analysed/210720 - ISO49+34 - projection_XY/unet_2022/project_XY_objects",
    "nuclei_path": "Secondary.csv",
}


kwargs = {
    "data_folder": "analysed/_2019_cellesce_unet_splineparameters_aligned/raw/projection_XY/",
    "nuclei_path": "Secondary.csv",
}


kwargs = {
    "data_folder": "analysed/_2019_cellesce_unet_splineparameters_aligned/raw/projection_XY/",
    "nuclei_path": "Secondary.csv",
}

kwargs_cellprofiler = {
    "data_folder": "analysed/210720 - ISO49+34 - projection_XY/unet_2022/project_XY_objects",
    "nuclei_path": "object_filteredNuclei.csv",
}


# kwargs_splinedist = {
#     "data_folder": "analysed/cellprofiler",
#     "nuclei_path": "objects_FilteredNuclei.csv",
# }


kwargs_splinedist = {
    "data_folder": "analysed/cellesce_splinedist_controlpoints",
    "nuclei_path": "Secondary.csv",
}

kwargs_splinedist = {
    "data_folder": "old_results/control_points",
    "nuclei_path": "objects_FilteredNuclei.csv",
}

# kwargs_cellprofiler = {
#     "data_folder": "analysed/cellprofiler/splinedist",
#     "nuclei_path": "objects_FilteredNuclei.csv",
# }


kwargs_cellprofiler = {
    "data_folder": "old_results/analysed/cellprofiler/splinedist",
    "nuclei_path": "objects_FilteredNuclei.csv",
}

# %%
# kwargs_cellprofiler = {
#     "data_folder": "analysed/cellprofiler/splinedist_32",
#     "nuclei_path": "objects_FilteredNuclei.csv",
# }

kwargs = kwargs_splinedist
kwargs = kwargs_cellprofiler


def save_csv(df, path):
    df.to_csv(metadata(path))
    return df


results_folder = f"results/merged"
pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)


def metadata(x):
    path = pathlib.Path(results_folder, x)
    print(path)
    return path


# %%
from sklearn.metrics.pairwise import euclidean_distances
from cellesce import Cellesce


def augment_at_theta(df, function, i, theta):
    return (
        df.apply(function, axis=1, theta=theta)
        .assign(augmentation=i, angle=theta)
        .set_index(["augmentation", "angle"], append=True)
    )


def dask_augment_df_alt(df, function, fold=0):
    #
    fold_sequence = np.append(np.array(0), np.random.uniform(0, 2 * np.pi, fold))
    return [
        delayed(augment_at_theta).apply(df=df, function=function, i=i, theta=theta)
        for i, theta in enumerate(fold_sequence)
    ]


def dask_augment_df(df, function, fold=0):
    #
    fold_sequence = np.append(np.array(0), np.random.uniform(0, 2 * np.pi, fold))
    return [
        delayed(df)
        .apply(function, axis=1, theta=theta)
        .assign(augmentation=i, angle=theta)
        .set_index(["augmentation", "angle"], append=True)
        for i, theta in enumerate(fold_sequence)
    ]


def rotate_control_points(series, theta=0):
    series_x, series_y = series.pipe(flat_series_to_dx_dy)
    series_x_prime = series_x * np.cos(theta) - np.array(series_y * np.sin(theta))
    series_y_prime = np.array(series_x * np.sin(theta)) + series_y * np.cos(theta)
    # series_prime = pd.concat([series_x_prime,series_y_prime],axis=1).assign(Angle=theta)
    series_prime = pd.concat([series_x_prime, series_y_prime])
    return series_prime.reindex(series.index, axis=1)


# import numba
# @numba.jit
def rotate_control_points_np(array, theta=0):
    odds = np.arange(0, len(array) - 1, 2)
    evens = np.arange(1, len(array), 2)
    x, y = array[odds], array[evens]
    x_prime = x * np.cos(theta) - np.array(y * np.sin(theta))
    y_prime = np.array(x * np.sin(theta)) + y * np.cos(theta)
    array_prime = array.copy()
    array_prime[odds] = x_prime
    array_prime[evens] = y_prime
    return array_prime


def align_coords_to_origin(series):
    # Series only now
    series_x, series_y = series.pipe(flat_series_to_dx_dy)
    series_x_prime = series_x - np.mean(series_x)
    series_y_prime = series_y - np.mean(series_y)
    series_prime = pd.concat([series_x_prime, series_y_prime], axis=0)
    return series_prime.reindex(series.index, axis=1)


def align_coords_to_origin_np(array):
    # Series only now
    odds = np.arange(0, len(array) - 1, 2)
    evens = np.arange(1, len(array), 2)

    x, y = array[odds], array[evens]

    x_prime = x - np.mean(x)
    y_prime = y - np.mean(y)

    array_prime = array.copy()

    array_prime[odds] = x_prime
    array_prime[evens] = y_prime

    return array_prime


def flat_series_to_dx_dy(series):
    # Series only now
    odds = np.arange(0, len(series.index) - 1, 2)
    evens = np.arange(1, len(series.index), 2)

    series_x = series[odds.astype(str)]
    series_y = series[evens.astype(str)]

    return series_x, series_y


def df_add_augmentation_index(df, index_name="augmentation"):
    return df.set_index(
        df.groupby(level=df.index.names).cumcount().rename(index_name), append=True
    )


def augment_distance_matrix(df, axis=0):
    return pd.concat(
        [
            # Numpy roll each row by i for each column
            df.transform(np.roll, 1, i, 0)
            for i in range(len(df_splinedist.columns))
        ],
        axis=axis,
    )


def augment_repeat(df, fold=1):
    return df.reindex(df_cellprofiler.index.repeat(fold))


df_splinedist = (
    Cellesce(**kwargs_splinedist)
    .get_data()
    .cellesce.preprocess()
    .cellesce.clean()
    .assign(Features="Spline")
    .set_index(["Features"], append=True)
    .apply(align_coords_to_origin_np, axis=1, raw=True)
    .sample(frac=1)
)
# %%

# %%
TEST_ROT =1
if (TEST_ROT):
    x = df_splinedist.iloc[:, np.arange(0, len(df_splinedist.columns) - 1, 2)]
    y = df_splinedist.iloc[:, np.arange(1, len(df_splinedist.columns), 2)]
    plt.scatter(x.iloc[0],y.iloc[0])
    plt.show()

    df_splinedist_rot = df_splinedist.apply(rotate_control_points_np, theta=-np.pi/2,axis=1, raw=True)

    x=df_splinedist_rot.iloc[:, np.arange(0, len(df_splinedist.columns) - 1, 2)]
    y=df_splinedist_rot.iloc[:, np.arange(1, len(df_splinedist.columns), 2)]
    plt.scatter(x.iloc[0],y.iloc[0])
    plt.show()
# %%
# # %%
# # %%
# # distogram.flatten()
# # distogram = sklearn.metrics.pairwise.euclidean_distances(df.iloc[[0]].T)
# df_sorted = df_splinedist.reindex(
#     np.array(sorted(df_splinedist.columns.astype(np.int))).astype(np.str), axis=1
# )
# df_dist = df_sorted.apply(
#     lambda x: euclidean_distances(np.array(x).reshape(-1, 1)).flatten(),
#     axis=1,
#     result_type="expand",
# )
# # %%
# euclid = euclidean_distances(np.array(df_sorted.iloc[0]).reshape(-1, 1))

# # %%

# df_dist = df_sorted.apply(
#     lambda x: euclidean_distances(np.array([x[0::2], x[1::2]]).T).flatten(),
#     axis=1,
#     result_type="expand",
# )


def df_to_distance_matrix(df):
    return (
        df.apply(
            lambda x: np.tril(euclidean_distances(np.array([x[0::2], x[1::2]]).T))
            .flatten()
            .flatten(),
            axis=1,
            result_type="expand",
        )
        .replace(0, np.nan)
        .dropna(axis=1)
    )


# df_dist.columns = df_dist.columns.astype(str)
# # df_dist = (df_sorted
# #             .apply(lambda x: euclidean_distances(np.array([x[0::2],x[1::2]]).T,[[0,0]])
# #             .flatten(),axis=1,result_type="expand"))
# df_splinedist = df_dist
# rows, features = df_splinedist.shape

df_cellprofiler = (
    Cellesce(**kwargs_cellprofiler)
    .get_data()
    .cellesce.preprocess()
    .cellesce.clean()
    .assign(Features="Cellprofiler")
    .set_index(["Features"], append=True)
    # .sample(32,axis=1,random_state=42)
)
df_cellprofiler.columns = df_cellprofiler.columns.str.replace("AreaShape_", "")
df = pd.concat([df_cellprofiler, df_splinedist])


pca_spline = PCA(n_components=0.99).fit(df_splinedist)
pca_cellprofiler = PCA(n_components=0.99).fit(df_cellprofiler)

# pd.DataFrame(pca_spline.explained_variance_, columns=["Explained Variance"]).assign(
#     **{
#         "Features": "Control points",
#         "Principal Component": np.arange(0, len(pca_spline.explained_variance_)),
#     }
# )


exp_var = pd.concat(
    [
        pd.DataFrame(
            pca_spline.explained_variance_, columns=["Explained Variance"]
        ).assign(
            **{
                "Features": "Control points",
                "Principal Component": np.arange(
                    0, len(pca_spline.explained_variance_)
                ),
            }
        ),
        pd.DataFrame(
            pca_cellprofiler.explained_variance_, columns=["Explained Variance"]
        ).assign(
            **{
                "Features": "Cellprofiler",
                "Principal Component": np.arange(
                    0, len(pca_cellprofiler.explained_variance_)
                ),
            }
        ),
    ]
)

pca_component = pd.concat(
    [
        pd.DataFrame(pca_spline.components_, columns=df_splinedist.columns).assign(
            **{
                "Features": "Control points",
                "Principal Component": np.arange(
                    0, len(pca_spline.explained_variance_)
                ),
            }
        ),
        pd.DataFrame(
            pca_cellprofiler.components_, columns=df_cellprofiler.columns
        ).assign(
            **{
                "Features": "Cellprofiler",
                "Principal Component": np.arange(
                    0, len(pca_cellprofiler.explained_variance_)
                ),
            }
        ),
    ]
).set_index(["Features", "Principal Component"])


sns.catplot(
    x="Principal Component",
    hue="Features",
    y="Explained Variance",
    data=exp_var,
    legend_out=True,
    kind="bar",
)
plt.savefig(metadata("pca.pdf"), bbox_inches="tight")
plt.show()

# %%
component_melt = pd.melt(
    pca_component,
    var_name="Feature",
    value_name="Component Magnitude",
    ignore_index=False,
).set_index(["Feature"], append=True)
plt.show()

important_features = (
    component_melt.transform(abs)
    .reset_index()
    .sort_values("Component Magnitude", ascending=False)
    .drop_duplicates(["Features", "Principal Component"])
    .sort_values("Principal Component")
)

# %%
# sns.catplot(
#     x="Principal Component",
#     y="Feature",
#     data=important_features.reset_index(),
#     col="Features",
#     sharey=False,
# )
# plt.show()
# %%

# sns.catplot(
#     col="Principal Component",
#     y="Feature",
#     x="Component Magnitude",
#     sharey=False,
#     data=component_melt.reset_index(),
#     row="Features",
#     height=12,
# )
# plt.show()
# %%
# df = df.iloc[:,random.sample(range(0, features), 32)]

print(
    f'Organoids: {df.cellesce.grouped_median("ObjectNumber").cellesce.simple_counts()}',
    f"Nuclei: {df.cellesce.simple_counts()}",
)

upper = np.nanmean(df.values.flatten()) + 2 * np.nanstd(df.values.flatten())
lower = np.nanmean(df.values.flatten()) - 2 * np.nanstd(df.values.flatten())


def df_to_fingerprints_facet(*args, **kwargs):

    data = kwargs.pop("data")
    data = data.drop([*args[2:]], 1)

    image = data.dropna(axis=1, how="all")

    rows, cols = image.shape
    median_height = 0.1
    gap_height = 0.15
    # median_rows = int(rows*median_height/100)
    image_rows_percent = 1 - (gap_height + median_height)
    one_percent = rows / image_rows_percent
    # print(one_percent,rows)
    gap_rows = int(gap_height * one_percent)
    median_rows = int(median_height * one_percent)

    # median_rows,gaps_rows=(rows,rows)
    finger_print = image.median(axis=0)

    finger_print_image = np.matlib.repmat(finger_print.values, median_rows, 1)
    all_data = np.vstack([image, np.full([gap_rows, cols], np.nan), finger_print_image])

    # fig,ax = plt.subplots(figsize=(5,3), dpi=150)
    # fig, ax = plt.figure(figsize=(5,3), dpi=150)
    plt.imshow(all_data, vmin=lower, vmax=upper, cmap="Spectral")
    # sns.heatmap(all_data,vmin=lower, vmax=upper, cmap="Spectral",interpolation='nearest')
    fig, ax = (plt.gcf(), plt.gca())
    ax.set(adjustable="box", aspect="auto", autoscale_on=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_facecolor("white")
    ax.grid(False)
    # fig.add_subplot(2,2,1)


sns.set()
g = sns.FacetGrid(
    df.reset_index(level=["Cell", "Features"]),
    col="Features",
    row="Cell",
    height=2,
    aspect=1.61,
    sharey=False,
    sharex=False,
)
cax = g.fig.add_axes([1.015, 0.13, 0.015, 0.8])
g.map_dataframe(
    df_to_fingerprints_facet,
    "Features",
    "Nuclei",
    "Features",
    "Cell",
    cmap="Spectral",
    cbar=True,
    vmax=upper,
    vmin=lower,
)
plt.colorbar(cax=cax)
plt.savefig(metadata("fingerprints.pdf"), bbox_inches="tight")
plt.show()
# %%


# df_distance_matrix = (df_splinedist
#                       .pipe(df_to_distance_matrix)
#                       .pipe(augment_distance_matrix,axis=1))


# df = pd.concat([df_cellprofiler, df_splinedist])

# df = pd.concat([df_cellprofiler,
#                 df_splinedist.pipe(df_to_distance_matrix)])


# df = pd.concat([df_cellprofiler
#                     .pipe(augment_repeat, fold=1),
#                 df_splinedist
#                     .pipe(df_to_distance_matrix)
#                     .pipe(augment_distance_matrix)
#                     .pipe(df_add_augmentation_index)])
# %%

# %%
# TODO


# for augmentation in np.linspace(1,200,10).astype(int):
def augment_df_dask(df, function, fold=0):
    #
    fold_sequence = np.append(np.array(0), np.random.uniform(0, 2 * np.pi, fold))
    index = df.index
    df_no_index = dd.from_pandas(df.reset_index(drop=True), npartitions=len(df))
    # print(df_no_index)

    return pd.concat(
        [
            df_no_index.apply(function, axis=1, theta=theta)
            .assign(augmentation=i, angle=theta)
            .compute()
            .set_index(index, append=True)
            for i, theta in enumerate(fold_sequence)
        ]
    ).set_index(["augmentation", "angle"], append=True)


# for augmentation in np.linspace(1,200,10).astype(int):
def angular_augment_df(df, function, fold=0):
    #
    fold_sequence = np.append(np.array(0), np.random.uniform(0, 2 * np.pi, fold))
    # print(df_no_index)

    return pd.concat(
        [
            df.apply(function, theta=0, axis=1, raw=True)
            .assign(augmentation=i, angle=0)
            .set_index(["augmentation", "angle"], append=True)
            for i, theta in enumerate(fold_sequence)
        ]
    )

def angular_augment_X_y(X,y, function=rotate_control_points_np, fold=0):
    X_aug = angular_augment_df(X, function, fold)
    y_aug = angular_augment_df(y, lambda x,theta:x, fold)
    return X_aug, y_aug

# %%

def feature_importances(df,augment=None):
    return ((df.cellesce.grouped_median("ObjectNumber")
        .dropna(axis=1)
        .cellesce.feature_importances(
            variable="Cell", kfolds=10, augment=augment
        )
    ).assign(Augmentation=augmentation))

def scoring(df,augment=None):
    return ((df.cellesce.grouped_median("ObjectNumber")
        .dropna(axis=1)
        .cellesce.get_scoring_df(
            variable="Cell", kfolds=10, augment=augment
        )
    ).assign(Augmentation=augmentation))

importance_list = []
scoring_list = []

for augmentation in [0,250,500,1000]:

    # for fold in range(1,6):

    # df = pd.concat(
    #     [
    #         df_cellprofiler.pipe(augment_repeat, fold=fold).pipe(df_add_augmentation_index),
    #         df_splinedist.pipe(augment_df, rotate_control_points, fold=fold).pipe(
    #             df_add_augmentation_index
    #         ),
    #     ]
    # )
    # df = pd.concat(
    #     [
    #         df_cellprofiler.pipe(df_add_augmentation_index),
    #         df_splinedist.pipe(augment_df, rotate_control_points, fold=augmentation)
    #         # .pipe(df_add_augmentation_index),
    #     ]
    # )
    # dask_spline_df = dd.from_pandas(df_splinedist.reset_index(),npartitions=1)

    # temp_df = df_splinedist.cellesce.grouped_median("ObjectNumber")
    # temp_df.loc[:,:] = 1
    # df = pd.concat(
    #     [
    #         # df_splinedist.pipe(rotate_control_points,fold=10)
    #         df_cellprofiler.pipe(df_add_augmentation_index),
    #         df_splinedist.pipe(augment_df, rotate_control_points_np, fold=augmentation),
    #         # .pipe(df_add_augmentation_index),
    #     ]
    # )
    # df = pd.concat(
    #     [
    #         # df_splinedist.pipe(rotate_control_points,fold=10)
    #         df_cellprofiler,
    #         df_splinedist
    #         # .pipe(df_add_augmentation_index),
    #     ]
    # )
    # df_cellprofiler_rep = df_cellprofiler.reindex(df_cellprofiler.index.repeat(3)).pipe(
    #     df_add_augmentation_index
    # )

    # df = df_cellprofiler_rep
    # feature_importance = df.groupby(level="Features").apply(
    #     lambda df: df.cellesce.grouped_median("ObjectNumber")
    #     .dropna(axis=1)
    #     .cellesce.feature_importances(variable="Cell",kfolds=5,groupby="augmentation")
    # ).assign(Augmentation=augmentation)
    
    spline_augment = lambda X,y: angular_augment_X_y(X,y, rotate_control_points_np, fold=augmentation)
    
              
    # df_splinedist.pipe(feature_importances,augment=spline_augment)
    
    # feature_importance_df = pd.concat(
    #     [df_cellprofiler.pipe(feature_importances),
    #      df_splinedist.pipe(feature_importances,augment=angular_augment_X_y)])

    feature_importance_df = pd.concat(
        [(df_cellprofiler
            .cellesce.grouped_median("ObjectNumber")
            .pipe(feature_importances)
            .assign(Features="CellProfiler")
            .set_index("Features", append=True)),
         (df_splinedist
            .cellesce.grouped_median("ObjectNumber")
            .pipe(feature_importances,augment=angular_augment_X_y)
            .assign(Features="SplineDist")
            .set_index("Features", append=True))
         ])


    scoring_df = pd.concat(
        [(df_cellprofiler
            .cellesce.grouped_median("ObjectNumber")
            .pipe(scoring)
            .assign(Features="CellProfiler")
            .set_index("Features", append=True)),
         (df_splinedist
            .cellesce.grouped_median("ObjectNumber")
            .pipe(scoring,augment=angular_augment_X_y)
            .assign(Features="SplineDist")
            .set_index("Features", append=True))
         ])

    importance_list.append(feature_importance_df)
    scoring_list.append(scoring_df)

    # feature_importance = (
    #     df.groupby(level="Features").apply(
    #         lambda df: df.cellesce.grouped_median("ObjectNumber")
    #         .dropna(axis=1)
    #         .cellesce.feature_importances(
    #             variable="Cell", kfolds=10, groupby="augmentation"
    #         )
    #     )
    # ).assign(Augmentation=augmentation)

    # scoring = (
    #     df.groupby(level="Features").apply(
    #         lambda df: df.cellesce.grouped_median("ObjectNumber")
    #         .dropna(axis=1)
    #         .cellesce.get_scoring_df(variable="Cell", kfolds=10, groupby="augmentation")
    #     )
    # ).assign(Augmentation=augmentation)


scoring_df = pd.concat(scoring_list)
importance_df = pd.concat(importance_list)


scoring_df_mean = scoring_df.groupby(
    ["Augmentation", "Metric", "Kind", "Variable"]
).mean()
scoring_df_var = scoring_df.groupby(
    ["Augmentation", "Metric", "Kind", "Variable"]
).var()
print("fin")


# %%
# data = scoring_df.reset_index("Features")
# data.to_csv("scoring_df.csv")
sns.lmplot(
    x="Augmentation",
    y="Score",
    col="Kind",
    row="Metric",
    hue="Features",
    fit_reg=False,
    sharey=False,
    sharex=False,
    x_ci="ci",
    x_bins=5,
    data=(scoring_df.reset_index("Features")),
)
plt.savefig(metadata("scoring.pdf"))
plt.show()

sns.catplot(
    y="Feature",
    x="Importance",
    col="Features",
    sharey=False,
    kind="bar",
    aspect=1 / 3,
    height=12.5,
    data=(importance_df.reset_index()),
)
plt.savefig(metadata("feature_importance.pdf"))
plt.show()


# %%
# TODO figure how to add after data splitting

# df = (
#     df_cellprofiler.pipe(augment_repeat, fold=fold)
#     .pipe(df_add_augmentation_index)
#     .cellesce.grouped_median("ObjectNumber")
#     .dropna(axis=1)
# )

# df = (df_splinedist
#     .pipe(augment_df, rotate_control_points, fold=5)
#     .pipe(df_add_augmentation_index))

# groupby = list(set(df.index.names)-{"augmentation"})

# y = df.reset_index()[["Cell"]].astype(str)
# X_train, X_test, y_train, y_test = df.cellesce.train_test_split(
#     variable="Cell", groupby=["augmentation"], seed=42)
# # (RandomForestClassifier()
# #         .fit(X_train,
# #              y_train)
# #         .score(X_test.xs(0,level="augmentation"),
# #                y_test.xs(0,level="augmentation")))

# score = (RandomForestClassifier()
#         .fit(X_train,y_train)
#         .score(X_test,y_test))
# print(f"fold {fold} score {score}")


# %%
sns.catplot(
    y="Feature",
    x="Importance",
    col="Features",
    sharey=False,
    kind="bar",
    aspect=1 / 3,
    height=12.5,
    data=(
        (
            df.groupby(level="Features").apply(
                lambda df: df.cellesce.grouped_median()
                .dropna(axis=1)
                .cellesce.feature_importances(variable="Cell")
            )
        )
        .reset_index()
        .pipe(save_csv, "importance_median_control_points.csv")
    ),
)

# plt.tight_layout()
plt.savefig(metadata("importance_median_control_points.pdf"))
plt.show()


# %% Spline importance statistical testing

# sample = -1

spline_importances = importance_df.xs("SplineDist", level="Features")["Importance"]

cellprofiler_importances = importance_df.xs("Cellprofiler", level="Features")[
    "Importance"
]

spline_H = scipy.stats.entropy(
    spline_importances, qk=np.ones_like(spline_importances) / len(spline_importances)
)
spline_H

cellprofiler_H = scipy.stats.entropy(
    cellprofiler_importances,
    qk=np.ones_like(cellprofiler_importances) / len(cellprofiler_importances),
)
cellprofiler_H

# scipy.stats.ks_2samp(cellprofiler_importances,np.ones_like(cellprofiler_importances)/len(cellprofiler_importances))

# scipy.stats.ks_2samp(spline_importances,np.ones_like(spline_importances)/len(spline_importances))

spline_test = scipy.stats.normaltest(importance_df.xs("Spline", level="Features"))
cellprofiler_test = scipy.stats.normaltest(
    importance_df.xs("Cellprofiler", level="Features")
)
print(f"Spline: {spline_H} Cellprofiler: {cellprofiler_H}")
print(f"Spline: {spline_test.pvalue[0]} Cellprofiler: {cellprofiler_test.pvalue[0]}")

# %%
# sns.barplot(
#     y="Feature", x="Cumulative Importance",
#     data=df.cellesce.feature_importances(variable="Cell").reset_index()
# )
# plt.tight_layout()


# %% Could do better with median per imagenumber
#     data=(
#         pd.concat(
#         [(
#             df.cellesce.grouped_median("ObjectNumber")
#             .cellesce.get_score_report("Cell")
#             .assign(**{"Population type": "Organoid"})
#             .set_index("Metric")
#             .loc[['f1-score', 'recall','precision']]
#             .reset_index()
#             .assign(Fold=fold)
#             .pipe(save_csv,"Cell_predictions_organoid.csv")
#             )
#         for fold in range(5)]
#     )),
# data =  pd.concat([
#         (pd.concat(
#         [
#             (
#                 (df.groupby(level="Features")
#                  .apply(lambda df:
#                     df.dropna(axis=1)
#                     .cellesce.get_score_report(variable="Cell")))
#                 .assign(**{"Population type": "Nuclei"})
#             ),
#             (
#                 (df.groupby(level="Features")
#                  .apply(lambda df:
#                     df.cellesce.grouped_median().dropna(axis=1)
#                     .cellesce.get_score_report(variable="Cell")))
#                 .assign(**{"Population type": "Organoid"})
#             ),
#         ])
#         .reset_index()
#         .set_index("Metric")
#         .loc[['f1-score', 'recall','precision']])
#         .reset_index()
#         .assign(Fold=fold) for fold in range(5)
#         ]).pipe(save_csv,"Cell_predictions_image_vs_nuclei.csv")
# %%


# No augmentation
# df = pd.concat([df_cellprofiler, df_splinedist])

# Distance matrix method
# df = pd.concat([df_cellprofiler, df_splinedist.pipe(df_to_distance_matrix)])
# df = pd.concat(
#     [
#         df_cellprofiler.reindex(df_cellprofiler.index.repeat(100)).pipe(
#             df_add_augmentation_index
#         ),
#         df_splinedist.pipe(augment_df, rotate_control_points, fold=100).pipe(
#             df_add_augmentation_index
#         ),
#     ]
# )


# df_cellprofiler_rep = df_cellprofiler.reindex(df_cellprofiler.index.repeat(100)).pipe(df_add_augmentation_index)
# df_cellprofiler_rep.pipe(df_add_augmentation_index)

# pd.MultiIndex.from_arrays([df.index, df.groupby(level=0).cumcount()],
#                                      names=(list(df.index.names) + ["Augment"]))
# options = {
#     "raw": pd.concat([df_cellprofiler, df_splinedist]),
#     "distance_matrix": pd.concat(
#         [df_cellprofiler, df_splinedist.pipe(df_to_distance_matrix)]
#     ),
#     "angular_augment": pd.concat(
#         [
#             df_cellprofiler,
#             df_splinedist.pipe(augment_df, rotate_control_points, fold=10),
#         ]
#     ),
# }

# df = options["angular_augment"]


def get_score_report_per(df, level="Features"):
    return (
        df.groupby(level="Features")
        .apply(
            lambda df: df.cellesce.grouped_median()
            .dropna(axis=1)
            .cellesce.get_score_report(variable="Cell")
        )
        .reset_index()
        .set_index("Metric")
        .loc[["f1-score", "recall", "precision"]]
        .reset_index()
    )


plot = (
    sns.catplot(
        aspect=1.2,
        height=3,
        x="Features",
        y="Score",
        col="Metric",
        row="Kind",
        # ci=None,
        data=df.pipe(get_score_report_per, "Features"),
        sharey=False,
        kind="bar",
    )
    .set_xticklabels(rotation=45)
    .set(ylim=(0, 1))
)

# %%


df = pd.concat([df_cellprofiler, df_splinedist])


data_list = []
for fold in tqdm(range(5)):
    print(f"Fold {fold}")
    df_temp = (
        pd.concat(
            [
                (
                    (
                        df.groupby(level="Features").apply(
                            lambda df: df.dropna(axis=1).cellesce.get_score_report(
                                variable="Cell"
                            )
                        )
                    ).assign(**{"Population type": "Nuclei"})
                ),
                (
                    (
                        df.groupby(level="Features").apply(
                            lambda df: df.cellesce.grouped_median()
                            .dropna(axis=1)
                            .cellesce.get_score_report(variable="Cell")
                        )
                    ).assign(**{"Population type": "Organoid"})
                ),
            ]
        )
        .reset_index()
        .set_index("Metric")
        .loc[["f1-score", "recall", "precision"]]
        .reset_index()
        .assign(Fold=fold)
    )
    data_list.append(df_temp)

data = pd.concat(data_list).pipe(save_csv, "Cell_predictions_image_vs_nuclei.csv")
# %%
plot = (
    sns.catplot(
        aspect=1.2,
        height=3,
        x="Kind",
        y="Score",
        col="Metric",
        row="Features",
        # ci=None,
        hue="Population type",
        data=data,
        sharey=False,
        kind="bar",
    )
    .set_xticklabels(rotation=45)
    .set(ylim=(0, 1))
)
# plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata("Cell_predictions_image_vs_nuclei.pdf"))
plt.show()

plot = (
    sns.catplot(
        aspect=1,
        height=3,
        x="Features",
        y="Score",
        col="Metric",
        hue="Kind",
        # ci=None,
        data=data.set_index("Population type").loc["Organoid"],
        sharey=False,
        kind="bar",
    )
    .set_xticklabels(rotation=45)
    .set(ylim=(0, 1))
)
# plt.tight_layout()
if SAVE_FIG:
    plt.savefig(metadata("Cell_predictions_organoid.pdf"), bbox_inches="tight")
plt.show()

# %%


# df_temp = ((df.groupby(level="Features")
#                 .apply(lambda df:
#                     df.cellesce.grouped_median().dropna(axis=1)
#                     .cellesce.get_score_report(variable="Cell")))
#                 .assign(**{"Population type": "Organoid"})
#         .reset_index()
#         .set_index("Metric")
#         .loc[['f1-score', 'recall','precision']]
#         .reset_index())
# data_list.append(df_temp)
