# report: "report/workflow.rst"
from snakemake.remote import AUTO
from snakemake.remote.FTP import RemoteProvider as FTPRemoteProvider
import pandas as pd
FTP = FTPRemoteProvider(username="bsftp", password="bsftp1")
ftp_path = "ftp-private.ebi.ac.uk/b6/58f8c7-0d88-424c-96bd-63d97210703c-a408"
# ftp_path = "ftp-private.ebi.ac.uk/b6/58f8c7-0d88-424c-96bd-63d97210703c-a408/_2019_cellesce_uncropped/{image}.tif"
# print(FTP)
print(FTP.glob_wildcards(ftp_path))

# import pandas as pd
# FILE = "in/Stardist/Test - Images/cell migration R1 - Position 58_XY1562686154_Z0_T00_C1-image76.tif"

# FILE_OUT_PNG = "out/out.png"
# FILE_OUT_QUALITY = "quality.txt"
# FILE_OUT_QUALITY = "out/quality"

CSV_VARIANTS=["FilteredNuclei","Image","IdentifySecondaryObjects","nuclei_objects"]
FEATURE_INCLUSIONS=["all","objects"]

# CELLPROFILER_FILES = ["all_Experiment","all_FilteredNuclei","all_Image","all_IdentifySecondaryObjects","all_nuclei_objects",
# "objects_Experiment","objects_FilteredNuclei","objects_Image","objects_IdentifySecondaryObjects","objects_nuclei_objects"]
CELLPROFILER_FILES = ["all_Experiment"]

DATA_IN = "data"
IMAGES_IN_DIR = DATA_IN+"/cellesce_2d"

IMAGES_DIR = DATA_IN+"/Stardist/Training - Images"
MASKS_DIR = DATA_IN+"/Stardist/Training - Masks"

MODEL_OUT = "analysed/models"
# INFERENCE_IMAGES_IN, = glob_wildcards("in/Stardist/Test - Images/{input_images}.tif")
IMAGES_IN, = glob_wildcards("{input_images}.tif")

EXT = ".tif"

MODELS=["stardist","splinedist","unet"]
MODELS=["stardist"]
MODELS=["stardist","unet"]

# /b6/58f8c7-0d88-424c-96bd-63d97210703c-a408
# All is a special rule that takes in the last output of the pipeline

images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")


def aggregate_input_stardist(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
    # print(images_glob)
    # return images_glob
    # return expand("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif", images_in=images_glob)
    return expand("analysed/stardist_inference/{images_in}_labels.png", images_in=images_glob)


rule all:
	input:
		MODEL_OUT,
        # DATA_IN,
        IMAGES_IN_DIR,
        # "analysed/cellprofiler/stardist",
        # "analysed/cellprofiler/stardist/test.csv",
        # "analysed/cellprofiler/unet/test.csv",
        # expand("analysed/cellprofiler/unet/{feature_inclusions}_{csv_variants}.csv",
        #         feature_inclusions=["all","objects"],
        #         csv_variants=["Experiment","FilteredNuclei","Image","IdentifySecondaryObjects","nuclei_objects"])
        # expand("analysed/cellprofiler/unet/{feature_inclusions}_{csv_variants}.csv",
        #         feature_inclusions="all",
        #         csv_variants="nuclei_objects",
        #         ),
        # expand("analysed/cellprofiler/unet/{images_in}/{feature_inclusions}_{csv_variants}.csv",
        #         feature_inclusions="all",
        #         csv_variants="nuclei_objects",
        #         images_in=images_glob
        #         )
        # expand("analysed/cellprofiler/unet/{cellprofiler_files}.csv",
        #         cellprofiler_files="all_nuclei_objects"
        #         ),
        expand("analysed/cellprofiler/unet/{images_in}/{cellprofiler_files}.csv",
                        images_in=images_glob,
                        cellprofiler_files="all_nuclei_objects",
                        ),
        expand("analysed/cellprofiler/{cellprofiler_files}.csv",
                        cellprofiler_files="all_nuclei_objects"
                ),
        # expand("analysed/cellprofiler/unet/{cellprofiler_files}.csv",
        #         csv_variants=CSV_VARIANTS,
        #         ),
        # "analysed/cellprofiler/unet/test.csv",
        # "analysed/cellprofiler/unet/all_Experiment.csv",
        #  "analysed/cellprofiler/unet/all_Experiment.csv"
        # expand("analysed/cellprofiler/{model}/test.csv",model=MODELS),
        # "analysed/cellprofiler/unet/test.csv"
        # aggregate_input_unet,
        # expand("analysed/cellprofiler/stardist/test.csv"),
        # expand("analysed/cellprofiler/stardist")
        # expand("{input_images}.tif", input_images=IMAGES_IN)
        # "analysed/test.csv",
        # aggregate_input
        


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
        images_in="data/cellesce_2d/{images_in}/projection_XY_16_bit.tif"
    output:
       thumb="analysed/stardist_inference/{images_in}_thumb.png",
       labels="analysed/stardist_inference/{images_in}_labels.png"
    conda:
        "stardist/environment.yaml"
    params:
        script="stardist/infer.py"
    shell:
        "python \
           {params.script} \
            --image_in='{input.images_in}' \
            --model_path='{input.model}' \
            --figure_out='{output.thumb}' \
            --labels='{output.labels}' \
        "


rule unet_inference:
    input:
        model=MODEL_OUT,
        image_dir=IMAGES_IN_DIR,
        images_in="data/cellesce_2d/{images_in}/projection_XY_16_bit.tif"
    output:
    #    thumb="analysed/unet_inference/{images_in}_thumb.png",
       labels="analysed/unet_inference/{images_in}_labels.tif"
    resources:
        mem_mb=16000
    # threads: 1
    conda:
        "unet/environment.yaml"
    params:
        script="unet/infer.py"
    shell:
        "python \
           {params.script} \
            --image_in='{input.images_in}' \
            --labels='{output.labels}' \
        "

# images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")

# print(images_glob)

rule cellprofiler_csv_stardist:
    input:
        model=MODEL_OUT,
        # "analysed/stardist_inference/{images_in}.tif"
        # expand("analysed/stardist_inference/{images_in}.tif", images_in=images_glob)
        agg=aggregate_input_stardist
    output:
        csv_dir=directory("analysed/cellprofiler/stardist"),
        csv="analysed/cellprofiler/stardist/test.csv"
        # out=aggregate_input
    params:
        cp_config="cellprofiler/instance_cp4.cpproj"
    conda:
        "cellprofiler/environment.yaml"
    shell:
        # "touch {output.csv}" 
        # "bash  touch {output.csv}
        # mkdir {output.csv_dir} 
        # touch {output.csv} \
        "cellprofiler \
         -c -r -p '{params.cp_config}' \
        -i '{input}' \
        -o '{output}'' \
        "
# Use with rules

# all_Experiment="analysed/cellprofiler/unet/{images_in}/all_Experiment.csv"
# all_FilteredNuclei="analysed/cellprofiler/unet/{images_in}/all_FilteredNuclei.csv"
# all_Image="analysed/cellprofiler/unet/{images_in}/all_Image.csv"
# all_nuclei_objects="analysed/cellprofiler/unet/{images_in}/all_nuclei_objects.csv"
# objects_FilteredNuclei="analysed/cellprofiler/unet/{images_in}/objects_FilteredNuclei.csv"
# objects_Experiment="analysed/cellprofiler/unet/{images_in}/objects_Experiment.csv"
# objects_IdentifySecondaryObjects="analysed/cellprofiler/unet/{images_in}/objects_IdentifySecondaryObjects.csv"
# objects_Image="analysed/cellprofiler/unet/{images_in}/objects_Image.csv"
# objects_nuclei_objects="analysed/cellprofiler/unet/{images_in}/all_Experobjects_nuclei_objectsiment.csv"


checkpoint cellprofiler_csv_unet:
    input:
        model=MODEL_OUT,
        # "analysed/stardist_inference/{images_in}.tif"
        # expand("analysed/stardist_inference/{images_in}.tif", images_in=images_glob)
        # agg=aggregate_input_unet,
        # agg=expand("analysed/unet_inference/{images_in}_labels.tif", images_in=images_glob),
        image_in="analysed/unet_inference/{images_in}_labels.tif"
    output:
        # file_list = temp("analysed/unet_inference/{images_in}_file_list.txt"),
        # expand("'analysed/cellprofiler/unet/{images_in}/{feature_inclusions}_{csv_variants}.csv'",
        #     feature_inclusions=FEATURE_INCLUSIONS,
        #     csv_variants=CSV_VARIANTS,
        #     allow_missing=True)
        # "analysed/cellprofiler/unet/{images_in}/{feature_inclusions}_{csv_variants}.csv",
        # csv_dir=directory("analysed/cellprofiler/unet/{images_in}"),
        # "analysed/cellprofiler/unet/{images_in}/{feature_inclusions}_{csv_variants}.csv"
        "analysed/cellprofiler/unet/{images_in}/{cellprofiler_files}.csv"
    params:
        # csv_dir=directory("analysed/cellprofiler/unet/{images_in}"),
        file_list=temp("analysed/unet_inference/{images_in}_file_list.txt"),
        cp_config="cellprofiler/unet_cp4.cpproj",
        csv_dir=directory("analysed/cellprofiler/unet/{images_in}"),
        # csv_dir="analysed/cellprofiler/unet/{images_in}"
    # container:
        # "docker://cellprofiler/cellprofiler:4.2.1"
    conda:
        "cellprofiler/environment.yaml"
    shell:
        """
        echo "'{input.image_in}'" >> '{params.file_list}' && \
        cellprofiler \
        --run-headless \
        -c -r -o '{params.csv_dir}' \
        --pipeline '{params.cp_config}' \
        --file-list '{params.file_list}' \
        --log-level DEBUG
        """


# rule cellprofiler_csv_unet_temp_rule:
#     input:
#         expand("analysed/cellprofiler/unet/{images_in}/{cellprofiler_files}.csv",
#                 images_in=images_glob,
#                 cellprofiler_files=["objects_FilteredNuclei"])
#     output:
#          directory("analysed/cellprofiler/unet/{images_in}")

def aggregate_input_unet(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    # output = checkpoints.get_image_data.get(**wildcards)
    images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
    # return expand("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif", images_in=images_glob)
    return expand(
        "analysed/cellprofiler/unet/{images_in}",
            images_in=images_glob)
    # return expand(
    #     "analysed/cellprofiler/unet/{images_in}/{feature_inclusions}_{csv_variants}.csv",
    #         images_in=images_glob,
    #         allow_missing=True)
    # return expand("analysed/cellprofiler/unet/{images_in}",images_in=images_glob)
    # return expand("analysed/cellprofiler/unet/{images_in}/{feature_inclusions}_{csv_variants}.csv",
    #         images_in=images_glob,
    #         feature_inclusions=FEATURE_INCLUSIONS,
    #         csv_variants=CSV_VARIANTS,
    #         )

# def aggregate_input_unet_dir(wildcards):
#     checkpoints.get_image_data.get(**wildcards)
#     # output = checkpoints.get_image_data.get(**wildcards)
#     images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
#     # return expand("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif", images_in=images_glob)
#     return expand(
#         "analysed/cellprofiler/unet/{images_in}/",
#             images_in=images_glob)

# def images_glob(wildcards):
#     checkpoints.get_image_data.get()
#     images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
#     return images_glob


# def aggregate_input_cp_dirs(wildcards):
#     checkpoints.get_image_data.get(**wildcards)
#     # output = checkpoints.get_image_data.get(**wildcards)
#     images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
#     return expand(
#         "analysed/cellprofiler/unet/{images_in}",
#             images_in=images_glob,
#         )

def aggregate_input_cp_csvs(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    # output = checkpoints.get_image_data.get(**wildcards)
    # images_in,feature_inclusions,csv_variants, = glob_wildcards("analysed/cellprofiler/unet/{images_in}/{feature_inclusions}_{csv_variants}.csv")
    images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
    # return expand("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif", images_in=images_glob)
    return expand("analysed/cellprofiler/unet/{images_in}/{feature_inclusions}_{csv_variants}.csv",
                images_in=images_glob,
                allow_missing=True)


    
    # images_in, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
    # return expand(
    #     # "analysed/cellprofiler/unet/{images_in}",
    #     "analysed/cellprofiler/unet/{images_in}/all_Experiment.csv",
    #         images_in=images_in,
    #         )
    # return expand(
    #     # "analysed/cellprofiler/unet/{images_in}",
    #     "analysed/cellprofiler/unet/{images_in}/{{cellprofiler_files}}.csv",
    #         images_in=images_in,
    #         allow_missing=True
    #         )
            # allow_missing=True)

images_in, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
# images_in,cellprofiler_files = glob_wildcards("analysed/cellprofiler/unet/{images_in}/{cellprofiler_files}.csv")
# print(images_in)
# print(images_in)
# print(list(set(cellprofiler_files)))


# def aggregate_input_cp_csvs_dir(wildcards):
#     checkpoints.cellprofiler_csv_unet.get(**wildcards)
#     images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
#     return expand("analysed/cellprofiler/unet/{images_in}",images_in=images_glob)


rule cellprofiler_unet_merge:
    input:
        # agg=aggregate_input_unet,
        # aggregate_input_cp_dirs,
        # agg_csvs=aggregate_input_cp_csvs,
        # expand("analysed/cellprofiler/unet/{images_in}",images_in=images_in),
        # csvs=expand("analysed/cellprofiler/unet/{images_in}/{{cellprofiler_files}}.csv",images_in=images_in),
        ["analysed/cellprofiler/unet/{images_in}/{{cellprofiler_files}}.csv".format(images_in=images_in) for images_in in images_in]
        # csvs_dir=aggregate_input_cp_csvs_dir
        # agg_csvs="analysed/cellprofiler/unet/{images_in}/{cellprofiler_files}.csv"
        # csv_in ="analysed/cellprofiler/unet/{images_in}/{cellprofiler_files}.csv",
        # csv_in = expand(
        # # "analysed/cellprofiler/unet/{images_in}",
        #     "analysed/cellprofiler/unet/{images_in}/{{cellprofiler_files}}.csv",
        #     # zip,
        #     images_in=images_glob,
        #     # cellprofiler_files="all_Experiment",
        #     allow_missing=True
        #     )
        # agg_all="analysed/cellprofiler/unet/{images_in}/{feature_inclusions}_{csv_variants}.csv"
        # dir_in=aggregate_input_unet_dir
        # files=expand("analysed/cellprofiler/unet/{images_in}/{feature_inclusions}_{csv_variants}.csv",images_in=images_glob)
        # all_Experiment=expand("analysed/cellprofiler/unet/{images_in}/all_Experiment.csv", images_in=glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif"))
    # conda:
    #     "enviroment.yaml"
    params:
        # file_path = lambda wildcards, input: f'{input}/{wildcards.feature_inclusions}_{wildcards.csv_variants}.csv',
        # files = lambda wildcards, input: [s + f'/{wildcards.feature_inclusions}_{wildcards.csv_variants}.csv' for s in input],
    output:
        # csv="analysed/cellprofiler/unet/test.csv",
        "analysed/cellprofiler/{cellprofiler_files}.csv",
    shell:
        "echo output"
    # run:
        # print(params.files)
    # run:
    #     # print(input.split(",")[0])
    #     # files = pd.Series(input)+f"/{wildcards.feature_inclusions}_{wildcards.csv_variants}.csv"
    #     # print(files)
    #     # import pandas as pd
    #     # file_name = f'/{wildcards.feature_inclusions}_{wildcards.csv_variants}.csv'
    #     # files = [s + file_name for s in input]
    #     # # print(files[-1])
    #     # # print(output)
    #     # # print(files)
    #     # # dfs = [pd.read_csv(files[0])]
    #     # print(files[0])
    #     # print(output)
    #     print(params.files)
    #     try:
    #         df = pd.concat(map(pd.read_csv, params.files), ignore_index=True)
    #     except Exception as e:
    #         print(e)
    #     print(df)
    #     df = pd.concat((pd.read_csv(f) for f in params.files), ignore_index=True)
    #     # dfs = (pd.read_csv(f) for f in files)
    #     # dfs = list(map(pd.read_csv, files))
    #     # print(dfs)
    #     print(output)
    #     # df = pd.concat(dfs, ignore_index=True)
    #     print(df)
        # df.to_csv(output)
    # shell:
        # """
        # touch {output} && echo {input}
        # """
# rule cellprofiler_csv_compile:
#     input:
#         "stardist_out/{images_in}.csv"
#     script:
#         pd.