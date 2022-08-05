# Pipeline and analysis tools for nuclear phenotyping

This repository is designed for analysing images of nuclei using deep learning
However, it is specialised for analysing 2D max-Z projections of organoids, though it may have general usage outside of this specific task.

This does contains a full snakemake pipeline to:

- Train UNet and *Dist models
- Run inference using models on Cellesce data
- Convert inference images to per-nuclei features
- Compile nuclei into a csv

Add secrets
    
    set -o allexport; source secrets.env;set +o allexport

Install env

    make install.snakemake.env

Test Snakemake

    snakemake --dry-run

Produce graphs

    python splines.py

TODO:

- Get automatic zenodo uploading working
- Seperate the UNet package into it's own git repo
- Add Figures
