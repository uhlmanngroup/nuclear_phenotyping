# report: "report/workflow.rst"


from snakemake.remote import AUTO
from snakemake.remote.FTP import RemoteProvider as FTPRemoteProvider
import pandas as pd
import os
import dask.dataframe as dd
import pandas as pd


# # df = pd.concat(map(pd.read_csv, input.files_in), ignore_index=True)
# FTP = FTPRemoteProvider(username="bsftp", password="bsftp1")
# ftp_path = "***REMOVED***"
# # ftp_path = "***REMOVED***/_2019_cellesce_uncropped/{image}.tif"
# # print(FTP)
# print(FTP.glob_wildcards(ftp_path))

# import pandas as pd
# FILE = "in/Stardist/Test - Images/cell migration R1 - Position 58_XY1562686154_Z0_T00_C1-image76.tif"

# FILE_OUT_PNG = "out/out.png"
# FILE_OUT_QUALITY = "quality.txt"
# FILE_OUT_QUALITY = "out/quality"

CSV_VARIANTS=["FilteredNuclei","Image","nuclei_objects"]
FEATURE_INCLUSIONS=["all","objects"]
# Filename prefix in: unet_cp4_3_class.cpproj

CELLPROFILER_FILES = ["all_Experiment","all_FilteredNuclei","all_Image","all_IdentifySecondaryObjects","all_nuclei_objects",
"objects_Experiment","objects_FilteredNuclei","objects_Image","objects_IdentifySecondaryObjects","objects_nuclei_objects"]
# CELLPROFILER_FILES = ["all_Experiment"]

DATA_IN = "data"
IMAGES_IN_DIR = DATA_IN+"/cellesce_2d"

IMAGES_DIR = DATA_IN+"/Stardist/Training - Images"
MASKS_DIR = DATA_IN+"/Stardist/Training - Masks"

MODEL_OUT = "analysed/models"
# INFERENCE_IMAGES_IN, = glob_wildcards("in/Stardist/Test - Images/{input_images}.tif")
IMAGES_IN, = glob_wildcards("{input_images}.tif")

EXT = ".tif"

MODELS=["stardist","splinedist","unet"]
# MODELS=["stardist"]
MODELS=["stardist","unet"]
MODELS=["splinedist","unet"]

inference_segmentation_config={
            "splinedist":"cellprofiler/instance_cp4.cpproj",
            "stardist":"cellprofiler/instance_cp4.cpproj",
            "unet":"cellprofiler/unet_cp4_3_class.cpproj",
            }

# /b6/58f8c7-0d88-424c-96bd-63d97210703c-a408
# All is a special rule that takes in the last output of the pipeline

images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
# images_glob,cellprofiler_files = glob_wildcards("analysed/cellprofiler/unet/{images_in}/{cellprofiler_files}.csv")
images_in, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
renamed_images_glob, = glob_wildcards("analysed/data/{images}/")

def aggregate_input_stardist(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    images_glob, = glob_wildcards("data/cellesce_2d/{images}/projection_XY_16_bit.tif")
    # print(images_glob)
    # return images_glob
    # return expand("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif", images_in=images_glob)
    return expand("analysed/stardist_inference/{images}/labels.png", images_in=images_glob)

# print(
#             expand("{feature_inclusions}_{csv_variants}.csv",
#             feature_inclusions=FEATURE_INCLUSIONS,
#             csv_variants=CSV_VARIANTS)
# )


# print(renamed_images_glob)


def aggregate_decompress_images(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    images_raw, = glob_wildcards(
        "data/cellesce_2d/{images_raw}/projection_XY_16_bit.tif")
    images = [i.replace('/','_').replace(' ','_') for i in images_raw]
    # checkpoints.move_data.get(images_raw=images_raw,images=images,**wildcards)
    return expand("analysed/data/images/temp/{images_raw}/projection_XY_16_bit.chkpt", images_raw=images_raw)


rule all:
	input:
		MODEL_OUT,
        # aggregate_decompress_images,
        # DATA_IN,
        IMAGES_IN_DIR,
        # "analysed/data/{images}",
        # expand("analysed/cellprofiler_{model}/{feature_inclusions}_{csv_variants}.csv",
        #     feature_inclusions=FEATURE_INCLUSIONS,
        #     csv_variants=CSV_VARIANTS,
        #     model=MODELS),       
        #  expand("analysed/cellprofiler/{images}/{model}/",
        #                 images=images,
        #                 model=MODELS),
        expand("analysed/results/{model}/{feature_inclusions}_{csv_variants}.csv",
            model=MODELS,
            feature_inclusions=FEATURE_INCLUSIONS,
            csv_variants=CSV_VARIANTS),
        # expand("analysed/cellprofiler/{images}/{model}/{feature_inclusions}_{csv_variants}.csv",
        #                 allow_missing=True,
        #                 images=images,
        #                 model=MODELS,
        #                 feature_inclusions=FEATURE_INCLUSIONS,
        #                 csv_variants=CSV_VARIANTS,
        #             )
        # "analysed/data",
        # expand("analysed/data/temp/{images_in}",images_in=images_glob)


rule get_training_data:
    # input:
    params:
        data_in=directory(DATA_IN)
    output:
        images_dir=directory(IMAGES_DIR),
        masks_dir=directory(MASKS_DIR)
    shell:
        "wget -r ftp://ftp.ebi.ac.uk/biostudies/nfs/S-BSST/666/S-BSST666/Files/ZeroCostDl4Mic/Stardist_v2/Stardist \
        --no-parent -nH --cut-dirs=8 -P {params.data_in}"

rule dist_training:
	input:
		images_dir=IMAGES_DIR,
		masks_dir=MASKS_DIR
	output:
	    directory(MODEL_OUT)
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
        data_folder=DATA_IN
    output:
        temp("temp.zip")
    shell:
        "wget {params.link} -O {output}"

checkpoint get_image_data:
    input:
        "temp.zip"
    params:
        data_folder=DATA_IN
    output:
        directory(IMAGES_IN_DIR)
    shell:
        "unzip {input} -d {params.data_folder}"

# Combine rules 
checkpoint move_data:
    input:
        "data/cellesce_2d/{images_raw}/projection_XY_16_bit.tif"
    output:
        # images = "analysed/data/{images}",
        checkpoint="analysed/data/images/temp/{images_raw}/projection_XY_16_bit.chkpt"
        # folder="analysed/data/images/"
    params:
        file_name=lambda wildcards: "analysed/data/images/"+(wildcards.images_raw).replace('/','_').replace(' ','_')+'.tif'
    shell:
        """
        cp -n '{input}' '{params.file_name}'
        touch '{output.checkpoint}'
        """

checkpoint confirm_data:
    input:
        aggregate_decompress_images
# def aggregate_decompress_images_raw(wildcards):
#     checkpoints.get_image_data.get(**wildcards)
#     images_raw, = glob_wildcards(
#         "data/cellesce_2d/{images_raw}/projection_XY_16_bit.tif")
#     return expand("data/cellesce_2d/{images_raw}/projection_XY_16_bit.tif",
#         images_raw=images_raw)

# checkpoint aggregate_decompress_images_raw:
#     input: aggregate_decompress_images_raw
#     output: temp("aggregate_decompress_images_raw.flag")
#     shell:'''
#     touch {output}
#     '''

# def aggregate_decompress_images(wildcards):
#     checkpoints.aggregate_decompress_images_raw.get(**wildcards)
#     return "analysed/data/"+(wildcards.images_in).replace('/','_').replace(' ','_')



# checkpoint aggregate_decompress_images:
#     input: aggregate_decompress_images
#     output: temp("aggregate_decompress_images.flag")
#     shell:'''
#     touch {output}
#     '''

# def aggregate_input(wildcards):
#     checkpoint_output = checkpoints.get_image_data.get().output[0]
#     print(wildcards)
#     return None
#     # return expand("prot_tables/{c}.prot_table", 
#     #     c=glob_wildcards(os.path.join(checkpoint_output, "{c}.gff")).c)

rule dist_inference:
    input:
        model=MODEL_OUT,
        image_dir=IMAGES_IN_DIR,
        images="analysed/data/images/{images}"
    output:
       thumb="analysed/stardist_inference/{images}_thumb.png",
       labels="analysed/stardist_inference/{images}_labels.png"
    conda:
        "stardist/environment.yaml"
    params:
        script="stardist/infer.py"
    shell:
        "python \
           {params.script} \
            --image_in='{input.images}' \
            --model_path='{input.model}' \
            --figure_out='{output.thumb}' \
            --labels='{output.labels}' \
        "


rule unet_inference:
    input:
        model=MODEL_OUT,
        image_dir=IMAGES_IN_DIR,
        images="analysed/data/images/{images}.tif"
    output:
    #    thumb="analysed/unet_inference/{images_in}_thumb.png",
        raw_image="analysed/unet_inference/{images}/raw.png",
        labels="analysed/unet_inference/{images}/labels.tif",
        background="analysed/unet_inference/{images}/background.png",
        foreground="analysed/unet_inference/{images}/foreground.png",
        boundary="analysed/unet_inference/{images}/boundary.png",
        folder=directory("analysed/unet_inference/{images}")
        # expand("analysed/unet_inference/{{images_in}}_{class}.png",
        #     )
    resources:
        mem_mb=64000
    threads: 64
    conda:
        "unet/environment.yaml"
    params:
        script="unet/infer.py"
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
        folder=directory("analysed/splinedist_inference/{images}/")
        # expand("analysed/unet_inference/{{images_in}}_{class}.png",
        #     )
    resources:
        mem_mb=64000
    threads: 64
    conda:
        "splinedist/environment.yaml"
    params:
        script="splinedist/infer.py",
        model_name="model_16_dsb2018"
        # model_path=
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

# print(images_glob)

# rule cellprofiler_csv_stardist:
#     input:
#         model=MODEL_OUT,
#         # "analysed/stardist_inference/{images_in}.tif"
#         # expand("analysed/stardist_inference/{images_in}.tif", images_in=images_glob)
#         agg=aggregate_input_stardist
#     output:
#         csv_dir=directory("analysed/cellprofiler/stardist"),
#         csv="analysed/cellprofiler/stardist/test.csv"
#         # out=aggregate_input
#     params:
#         cp_config="cellprofiler/instance_cp4.cpproj"
#     conda:
#         "cellprofiler/environment.yaml"
#     shell:
#         # "touch {output.csv}" 
#         # "bash  touch {output.csv}
#         # mkdir {output.csv_dir} 
#         # touch {output.csv} \
#         "cellprofiler \
#          -c -r -p '{params.cp_config}' \
#         -i '{input}' \
#         -o '{output}' \
#         "

rule cellprofiler_csv:
    input:
        image_in="analysed/{model}_inference/{images}/raw.png",
        # image_in="analysed/{model}_inference/{images_in}/labels.tif",
        # background="analysed/{model}_inference/{images_in}/background.png",
        # foreground="analysed/{model}_inference/{images_in}/foreground.png",
        # boundary="analysed/{model}_inference/{images_in}/boundary.png",
        folder="analysed/{model}_inference/{images}/",
    output:
        files=
            touch(expand("analysed/cellprofiler/{images}/{model}/{feature_inclusions}_{csv_variants}.csv",
                allow_missing=True,
                csv_variants=CSV_VARIANTS,
                feature_inclusions=FEATURE_INCLUSIONS,
            )),
        # folder = "analysed/cellprofiler/{images}/{model}/"
        folder = directory("analysed/cellprofiler/{images}/{model}/"),
    resources:
        mem_mb=2000
    params:
        cp_config=lambda wildcards: inference_segmentation_config[wildcards.model],
        # file_list=temp("analysed/cellprofiler/{model}/{images_in}/file_list.txt"),
        folder = directory("analysed/cellprofiler/{images}/{model}"),
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

# ruleorder: cellprofiler_csv_unet > cellprofiler_unet_merge

# images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")


def cellprofiler_merge_input(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    images_glob, = glob_wildcards("analysed/data/{images}/")
    # print(images_glob)
    # return images_glob
    # return expand("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif", images_in=images_glob)
    return expand("analysed/cellprofiler/{images}/{{model}}/",images=images_glob)

images_glob, = glob_wildcards("analysed/data/{images}/")
# rule cellprofiler_merge:
#     input:
#         # files =
#         #     expand("analysed/cellprofiler/{images}/{{model}}/{{feature_inclusions}}_{{csv_variants}}.csv",
#         #         allow_missing=True,
#         #         images=images_glob,
#         #     ),
#         folder = expand("analysed/cellprofiler/{images}/{{model}}/",
#                     allow_missing=True,
#                     images=renamed_images_glob),
#         # files_in = expand("analysed/cellprofiler/{images}/{{model}}/{{feature_inclusions}}_{{csv_variants}}.csv",
#         #         images=images_glob)
#         # folder = cellprofiler_merge_input
#         # folder = expand("analysed/cellprofiler/unet/{images_in}",images_in=images_glob),
#     params:
#         file_path = lambda wildcards, input: f'{input}/{wildcards.model}/{wildcards.feature_inclusions}_{wildcards.csv_variants}.csv',
#         files = lambda wildcards, input: [s + f'/{wildcards.feature_inclusions}_{wildcards.csv_variants}.csv' for s in input],
#     output:
#         csv="analysed/cellprofiler/{model}/{feature_inclusions}_{csv_variants}.csv"
#     run:
#         try:
#             df = pd.concat(map(pd.read_csv, params.files), ignore_index=True)
#             # df = pd.concat(map(pd.read_csv, input.files_in), ignore_index=True)
#             df.to_csv(output.csv)
#         except Exception as e:
#             print(e)

def cellprofiler_merge(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    images_raw, = glob_wildcards(
        "data/cellesce_2d/{images_raw}/projection_XY_16_bit.tif")
    checkpoints.confirm_data.get(**wildcards)
    images, = glob_wildcards("analysed/data/images/{images}.tif")
    return expand(
            "analysed/cellprofiler/{images}/{model}/{feature_inclusions}_{csv_variants}.csv",
            allow_missing=True,
            images=images,)
# images, = glob_wildcards("analysed/data/images/{images}.tif")

rule cellprofiler_merge:
    input:
        cellprofiler_merge
    output:
        csv="analysed/results/{model}/{feature_inclusions}_{csv_variants}.csv"
    resources:
        mem_mb=16000
    run:
        try:
            glob_string = f'analysed/cellprofiler/*cellesce*/{wildcards.model}/{wildcards.feature_inclusions}_{wildcards.csv_variants}.csv'
            print(glob_string)
            df = dd.read_csv(glob_string)
            # print(df)
            # df = pd.concat(map(pd.read_csv, input.files_in), ignore_index=True)
            df = df.compute()
            if "PathName_image" in df.columns:
                df["ImageNumber"] = pd.factorize(df["PathName_image"])[0]
            df.to_csv(output.csv,index=False)
            
        except Exception as e:
            print(e)

# rule control_points:
#     input:
#         folder = expand("analysed/cellprofiler/{images}/splinedist/",
#                     allow_missing=True,
#                     images=renamed_images_glob),
#     output:
#         csv="analysed/cellprofiler/splinedist/control_points.csv"