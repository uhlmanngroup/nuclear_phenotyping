rule convert_to_png:
    input:
        lif=expand(
            "{data_dir}/{folder}/{filename}{file_suffix}", allow_missing=True, **config
        )[0],
        # metadata="results/plast_cell/{filename}/metadata.json",
    conda:
        "../envs/convert_to_png.yaml",
    output:
        # image_dir=dynamic("results/plast_cell/{filename}"),
        png="results/plast_cell/{folder}/{filename}/i={i}_t={t}_z={z}_c={c}.png",
    script:
        "../scripts/convert_to_png.py"
