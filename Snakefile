# report: "report/workflow.rst"
from snakemake.remote.zenodo import RemoteProvider,
import os

from snakemake.remote import AUTO

# from snakemake.remote.FTP import RemoteProvider as FTPRemoteProvider


import os


# let Snakemake assert the presence of the required environment variable
envvars:
    "ZENODO_ACCESS_TOKEN",
    "BIOSTUDIES_FTP_SERVER",


zenodo_access_token = os.environ["ZENODO_ACCESS_TOKEN"]
deposition_id = 7267108

zenodo = RemoteProvider(deposition=deposition_id, access_token=zenodo_access_token)


CSV_VARIANTS = ["FilteredNuclei", "Image"]
FEATURE_INCLUSIONS = ["all", "objects"]

CELLPROFILER_FILES = [
    "all_Experiment",
    "all_FilteredNuclei",
    "all_Image",
    "all_IdentifySecondaryObjects",
    "all_nuclei_objects",
    "objects_Experiment",
    "objects_FilteredNuclei",
    "objects_Image",
    "objects_IdentifySecondaryObjects",
    "objects_nuclei_objects",
]

DATA_IN = "data"
IMAGES_IN_DIR = DATA_IN + "/cellesce_2d"

IMAGES_DIR = DATA_IN + "/Training - Images"
MASKS_DIR = DATA_IN + "/Stardist/Training - Masks"

MODEL_OUT = "analysed/models"
(IMAGES_IN,) = glob_wildcards("{input_images}.tif")

EXT = ".tif"

MODELS = ["stardist", "splinedist", "unet"]
# MODELS=["stardist"]
MODELS = ["stardist", "unet"]
MODELS = ["splinedist", "unet"]

inference_segmentation_config = {
    "splinedist": "cellprofiler/instance_cp4.cpproj",
    "stardist": "cellprofiler/instance_cp4.cpproj",
    "unet": "cellprofiler/unet_cp4_3_class.cpproj",
}


def aggregate_input_stardist(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    (images_glob,) = glob_wildcards(
        "data/cellesce_2d/{images}/projection_XY_16_bit.tif"
    )
    return expand(
        "analysed/stardist_inference/{images}/labels.png", images_in=images_glob)


def aggregate_decompress_images(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    (images_raw,) = glob_wildcards(
        "data/cellesce_2d/{images_raw}/projection_XY_16_bit.tif"
    )
    images = [i.replace("/", "_").replace(" ", "_") for i in images_raw]
    # checkpoints.move_data.get(images_raw=images_raw,images=images,**wildcards)
    return expand(
        "analysed/data/images/temp/{images_raw}/projection_XY_16_bit.chkpt",
images_raw=images_raw,
    )


rule all:
    input:
        # MODEL_OUT,
        aggregate_decompress_images,
        IMAGES_IN_DIR,
        expand(
            "results/{model}/{feature_inclusions}_{csv_variants}.csv",
            model=MODELS,
            feature_inclusions=FEATURE_INCLUSIONS,
            csv_variants=CSV_VARIANTS,
        ),
        zenodo.remote("results_csv.zip"),


rule get_training_data:
    # input:
    params:
        data_in=directory(DATA_IN),
    output:
        images_dir=directory(IMAGES_DIR),
        masks_dir=directory(MASKS_DIR),
    shell:
        "wget -r ftp://ftp.ebi.ac.uk/biostudies/nfs/S-BSST/666/S-BSST666/Files/ZeroCostDl4Mic/Stardist_v2/Stardist \
        --no-parent -nH --cut-dirs=8 -P {params.data_in}"


rule dist_training:
    input:
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
    output:
        directory(MODEL_OUT + "/stardist"),
    conda:
        "stardist/environment.yaml"
    # resources:
    #     nvidia_gpu=1
    shell:
        "python \
        stardist/training.py \
        --images_dir='{input.images_dir}'\
        --masks_dir='{input.masks_dir}'\
        --model_path='{output}'\
        --ext='.tif'\
        --epochs=50 \
        "


rule download_image_data:
    params:
        link="https://zenodo.org/record/6566910/files/cellesce_2d.zip?download=1",
        data_folder=DATA_IN,
    output:
        "raw_images.zip",
    shell:
        "wget {params.link} -O {output}"


checkpoint get_image_data:
    input:
        "raw_images.zip",
    params:
        data_folder=directory(DATA_IN),
    output:
        folder=directory(IMAGES_IN_DIR),
    shell:
        "unzip -o {input} -d {params.data_folder}"


checkpoint move_data:
    input:
        folder=IMAGES_IN_DIR,
        file="data/cellesce_2d/{images_raw}/projection_XY_16_bit.tif",
    output:
        checkpoint_dir = directory("analysed/data/images/temp/{images_raw}/"),
        checkpoint=touch("analysed/data/images/temp/{images_raw}/projection_XY_16_bit.chkpt"),
    params:
        file_name=lambda wildcards: "analysed/data/images/"
        + (wildcards.images_raw).replace("/", "_").replace(" ", "_")
        + ".tif",
    shell:
        """
        mkdir -p '{params.checkpoint_dir}'
        cp -n '{input.file}' '{params.file_name}'
        touch '{output.checkpoint}'
        """


checkpoint confirm_data:
    input:
        aggregate_decompress_images,


rule dist_inference:
    input:
        model=MODEL_OUT,
        image_dir=IMAGES_IN_DIR,
        images="analysed/data/images/{images}",
    output:
        thumb="analysed/stardist_inference/{images}_thumb.png",
        labels="analysed/stardist_inference/{images}_labels.png",
    conda:
        "stardist/environment.yaml"
    params:
        script="stardist/infer.py",
    shell:
        """
        python \
           {params.script} \
            --image_in='{input.images}' \
            --model_path='{input.model}' \
            --figure_out='{output.thumb}' \
            --labels='{output.labels}' \
        """


rule unet_inference:
    input:
        model=MODEL_OUT,
        image_dir=IMAGES_IN_DIR,
        images="analysed/data/images/{images}.tif",
    output:
        raw_image="analysed/unet_inference/{images}/raw.png",
        labels="analysed/unet_inference/{images}/labels.tif",
        background="analysed/unet_inference/{images}/background.png",
        foreground="analysed/unet_inference/{images}/foreground.png",
        boundary="analysed/unet_inference/{images}/boundary.png",
        folder=directory("analysed/unet_inference/{images}"),
    resources:
        mem_mb=64000,
    threads: 64
    conda:
        "unet/environment.yaml"
    params:
        script="unet/infer.py",
    shell:
        """
        python \
           {params.script} \
            --image_in='{input.images}' \
            --labels='{output.labels}' \
            --background_image='{output.background}' \
            --foreground_image='{output.foreground}' \
            --boundary_image='{output.boundary}' \
            --raw_image='{output.raw_image}'\
        """


rule splinedist_inference:
    input:
        model=MODEL_OUT,
        images="analysed/data/images/{images}.tif",
    output:
        #    thumb="analysed/unet_inference/{images_in}_thumb.png",
        raw_image="analysed/splinedist_inference/{images}/raw.png",
        instance="analysed/splinedist_inference/{images}/instance.png",
        control_points="analysed/splinedist_inference/{images}/control_points.csv",
        folder=directory("analysed/splinedist_inference/{images}/"),
        # expand("analysed/unet_inference/{{images_in}}_{class}.png",
        #     )
    resources:
        mem_mb=64000,
    threads: 64
    conda:
        "splinedist/environment.yaml"
    params:
        script="splinedist/infer.py",
        model_name="model_16_dsb2018",
    shell:
        """
        python \
           {params.script} \
            --image_in='{input.images}' \
            --instance='{output.instance}' \
            --control_points='{output.control_points}'\
            --raw_image='{output.raw_image}'\
            --model_path='{input.model}' \
            --model_name='{params.model_name}'\
        """


rule cellprofiler_csv:
    input:
        image_in="analysed/{model}_inference/{images}/raw.png",
        folder="analysed/{model}_inference/{images}/",
    output:
        files=expand(
            "analysed/cellprofiler/{images}/{model}/{feature_inclusions}_{csv_variants}.csv",
            allow_missing=True,
            feature_inclusions=FEATURE_INCLUSIONS,
            csv_variants=CSV_VARIANTS,
        ),
        folder=directory("analysed/cellprofiler/{images}/{model}/"),
    resources:
        mem_mb=2000,
    params:
        cp_config=lambda wildcards: inference_segmentation_config[wildcards.model],
        folder=directory("analysed/cellprofiler/{images}/{model}"),
        csv_dir=directory("analysed/{model}_inference/{images}"),
    # container:
    # "docker://cellprofiler/cellprofiler:4.2.1"
    conda:
        "cellprofiler/environment.yaml"
    shell:
        """
        cellprofiler \
        --run-headless \
        -c -r \
        -o '{output.folder}' \
        -i '{input.folder}' \
        --pipeline '{params.cp_config}' \
        --log-level DEBUG
        """


def cellprofiler_merge(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    (images_raw,) = glob_wildcards(
        "data/cellesce_2d/{images_raw}/projection_XY_16_bit.tif"
    )
    checkpoints.confirm_data.get(**wildcards)
    (images,) = glob_wildcards("analysed/data/images/{images}.tif")
    return expand(
        "analysed/cellprofiler/{images}/{model}/{feature_inclusions}_{csv_variants}.csv",
        allow_missing=True,
        images=images,
    )


rule cellprofiler_merge:
    input:
        checkpoint=aggregate_decompress_images,
        images_list=cellprofiler_merge,
    output:
        csv="results/{model}/{feature_inclusions}_{csv_variants}.csv",
    resources:
        mem_mb=16000,
    run:
        try:
            glob_string = input.images_list
            df = dd.read_csv(glob_string)

            df = df.compute()
            if "PathName_image" in df.columns:
                df["ImageNumber"] = pd.factorize(df["PathName_image"])[0]
            df.to_csv(output.csv, index=False)

        except Exception as e:
            print(e)


rule upload:
    input:
        expand(
            "results/{model}/{feature_inclusions}_{csv_variants}.csv",
            model=MODELS,
            feature_inclusions=FEATURE_INCLUSIONS,
            csv_variants=CSV_VARIANTS,
        ),
    params:
        # access_token=zenodo_access_token,
        # deposition_id=deposition_id, # replace with your actual deposition ID
    output:
        zip_file=temp("data/results_csv.zip"),
        remote=zenodo.remote("results_csv.zip"),
        # remote=zenodo.remote("zenodo://{access_token}/{deposition_id}/results_csv.zip"),
    shell:
        """
        zip {output.zip_file} {input}
        cp {output.remote} {output.zip_file}
        """


# rule control_points:
#     input:
#         folder = expand("analysed/cellprofiler/{images}/splinedist/",
#                     allow_missing=True,
#                     images=renamed_images_glob),
#     output:
#         csv="analysed/cellprofiler/splinedist/control_points.csv"
