rule convert_to_png:
    input:
        lif=expand(
            "{data_dir}/{folder}/{filename}{file_suffix}", allow_missing=True, **config
        ),
        # metadata="results/plast_cell/{filename}/metadata.json",
    output:
        # image_dir=dynamic("results/plast_cell/{filename}"),
        png="results/plast_cell/{folder}/{filename}/i={i}_t={t}_z={z}_c={c}.png",
    run:
        ims = pims.Bioformats(input.lif[0])
        ims.iter_axes = "ct"
        # frame = ims[int(i)]
        # breakpoint()
        im = PIL.Image.fromarray(ims[int(wildcards.i)])
        # im = im.convert('L') 
        # im = PIL.ImageOps.equalize(im, mask=None)

        im.save(output.png)
        print(f"Saving {output.png}")