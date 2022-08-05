SHELL := /bin/bash

zip.examples:
	zip -r examples analysed/splinedist_inference/_2019_cellesce_2019-05-16_190516_10.23.35_ISO49_Vemurafenib_0uM_Position_7_190516_10.23.35_Step_Size_+0.4_Wavelength_DAPI_452-45_Position_Position_7_ISO49_Vemurafenib_0uM analysed/splinedist_inference/_2019_cellesce_2019-04-02_190402_11.57.55_ISO34_Vemurafenib_0uM_190402_11.57.55_Step_Size_-0.4_Wavelength_DAPI_452-45_ISO34_Vemurafenib_0uM

zip.results:
	zip -r results analysed/cellprofiler/splinedist_32 analysed/cellprofiler/splinedist control_points

install.snakemake.env:
	mamba env create -f environment.yml --force

add.secrets:
	set -o allexport
	source secrets.env
	set +o allexport