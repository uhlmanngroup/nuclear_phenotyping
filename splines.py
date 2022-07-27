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

sns.set()
from cellesce import Cellesce

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
    "data_folder": "control_points",
    "nuclei_path": "objects_FilteredNuclei.csv",
}

kwargs_cellprofiler = {
    "data_folder": "analysed/cellprofiler/splinedist",
    "nuclei_path": "objects_FilteredNuclei.csv",
}


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

df_splinedist = (
    Cellesce(**kwargs_splinedist)
    .get_data()
    .cellesce.preprocess()
    .cellesce.clean()
    .assign(Features="Spline")
    .set_index(["Features"], append=True)
)
# %%


# %%
# %%
# distogram.flatten()
# distogram = sklearn.metrics.pairwise.euclidean_distances(df.iloc[[0]].T)
df_sorted = df_splinedist.reindex(
    np.array(sorted(df_splinedist.columns.astype(np.int))).astype(np.str), axis=1
)
df_dist = df_sorted.apply(
    lambda x: euclidean_distances(np.array(x).reshape(-1, 1)).flatten(),
    axis=1,
    result_type="expand",
)
df_dist = df_sorted.apply(
    lambda x: euclidean_distances(np.array([x[0::2], x[1::2]]).T).flatten(),
    axis=1,
    result_type="expand",
)

df_dist = (
    df_sorted.apply(
        lambda x: np.tril(euclidean_distances(np.array([x[0::2], x[1::2]]).T))
        .flatten()
        .flatten(),
        axis=1,
        result_type="expand",
    )
    .replace(0, np.nan)
    .dropna(axis=1)
)
df_dist.columns = df_dist.columns.astype(str)
# df_dist = (df_sorted
#             .apply(lambda x: euclidean_distances(np.array([x[0::2],x[1::2]]).T,[[0,0]])
#             .flatten(),axis=1,result_type="expand"))
df_splinedist = df_dist
rows, features = df_splinedist.shape
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

pd.DataFrame(pca_spline.explained_variance_, columns=["Explained Variance"]).assign(
    **{
        "Features": "Control points",
        "Principal Component":
            np.arange(0, len(pca_spline.explained_variance_)),
    }
)


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
plt.show()

# sns.catplot(
#     x="Principal Component",
#     y="Feature",
#     data=important_features.reset_index(),
#     col="Features",
#     sharey=False,
# )
# plt.show()


sns.catplot(
    col="Principal Component",
    y="Feature",
    x="Component Magnitude",
    sharey=False,
    data=component_melt.reset_index(),
    row="Features",
    height=12,
)
plt.show()
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
feature_importance = df.groupby(level="Features").apply(
    lambda df: df.cellesce.grouped_median()
    .dropna(axis=1)
    .cellesce.feature_importances(variable="Cell")
)


spline_test = scipy.stats.normaltest(feature_importance.xs("Spline", level="Features"))
cellprofiler_test = scipy.stats.normaltest(
    feature_importance.xs("Cellprofiler", level="Features")
)


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
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
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
