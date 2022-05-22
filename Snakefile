
# report: "report/workflow.rst"
from snakemake.remote import AUTO
from snakemake.remote.FTP import RemoteProvider as FTPRemoteProvider

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
        "analysed/cellprofiler/unet/test.csv",
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
    threads: 1
    conda:
        "unet/environment.yaml"
    params:
        script="unet/infer.py"
    shell:
        "python \
           {params.script} \
            --image_in='{input.images_in}' \
            --labels={output.labels} \
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

rule cellprofiler_csv_unet:
    input:
        model=MODEL_OUT,
        # "analysed/stardist_inference/{images_in}.tif"
        # expand("analysed/stardist_inference/{images_in}.tif", images_in=images_glob)
        # agg=aggregate_input_unet,
        # agg=expand("analysed/unet_inference/{images_in}_labels.tif", images_in=images_glob),
        image_in="analysed/unet_inference/{images_in}_labels.tif"
    output:
        csv_dir=directory("analysed/cellprofiler/unet/{images_in}"),
        file_list = temp("analysed/unet_inference/{images_in}_file_list.txt")
        # csv="analysed/cellprofiler/unet/test.csv"
        # out=aggregate_input
    params:
        cp_config="cellprofiler/unet_cp4.cpproj"
    # container:
        # "docker://cellprofiler/cellprofiler:4.2.1"
    conda:
        "cellprofiler/environment.yaml"
    shell:
        """
        echo "'{input.image_in}'" >> '{output.file_list}' && \
        cellprofiler \
        --run-headless \
        --pipeline '{params.cp_config}' \
        --file-list '{output.file_list}' \ 
        --output-directory '{output.csv_dir}' \
        --log-level DEBUG
        """

def aggregate_input_unet(wildcards):
    checkpoints.get_image_data.get(**wildcards)
    # output = checkpoints.get_image_data.get(**wildcards)
    images_glob, = glob_wildcards("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif")
    # return images_glob
    # print(images_glob)
    # return images_glob
    # return expand("data/cellesce_2d/{images_in}/projection_XY_16_bit.tif", images_in=images_glob)
    return expand("analysed/cellprofiler/unet/{images_in}", images_in=images_glob)


rule cellprofiler_unet_merge:
    input:
        agg=aggregate_input_unet,
        # csv_dir="analysed/cellprofiler/unet",
    output:
        csv="analysed/cellprofiler/unet/test.csv"
    shell:
        "touch {output}"
# rule cellprofiler_csv_compile:
#     input:
#         "stardist_out/{images_in}.csv"
#     script:
#         pd.