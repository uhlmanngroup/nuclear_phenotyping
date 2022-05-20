
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

MODEL_OUT = "models"
# INFERENCE_IMAGES_IN, = glob_wildcards("in/Stardist/Test - Images/{input_images}.tif")
IMAGES_IN, = glob_wildcards("{input_images}.tif")

EXT = ".tif"

# /b6/58f8c7-0d88-424c-96bd-63d97210703c-a408
# All is a special rule that takes in the last output of the pipeline
rule all:
	input:
		MODEL_OUT,
        DATA_IN,
        IMAGES_IN_DIR,
        expand("{input_images}.tif", input_images=IMAGES_IN)

rule download_image_data:
    params:
        link="https://zenodo.org/record/6566910/files/cellesce_2d.zip?download=1",
        data_folder=DATA_IN
    output:
        "temp.zip"
    shell:
        "wget {params.link} -O {output}"

rule get_image_data:
    input:
        "temp.zip"
    params:
        data_folder=DATA_IN
    output:
        directory(IMAGES_IN_DIR)
    shell:
        "unzip {input} -d {params.data_folder}"

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
	    "stardist.yaml"
	shell:
		"python \
            stardist/training.py \
            --images_dir='{input.images_dir}'\
            --masks_dir='{input.masks_dir}'\
            --model_path='{output}'\
            --ext='.tif'\
            --epochs=50 \
        "

rule dist_inference:
    input:
        model=MODEL_OUT,
        images_in=IMAGES_IN_DIR+"/{images_in}.tif"
    output:
       "analysed/stardist_inference/{images_in}.tif"
    conda:
        "stardist.yaml"
    shell:
        "python \
            stardist/infer.py \
            --image_in='{input.images_in}' \
            --model_path='{input.model}' \
            --figure_out='{output}' \
        "

rule cellprofiler_csv:
    input:
        "analysed/{images_in}.png"
    output:
        "analysed/{images_in}.csv"
    conda:
        "cellprofiler.yaml"
    shell:
        "cellprofiler -c -r -p cellprofiler/unet_cp4.cpproj.cppipe -o test.csv"


# rule cellprofiler_csv_compile:
#     input:
#         "stardist_out/{images_in}.csv"
#     script:
#         pd.