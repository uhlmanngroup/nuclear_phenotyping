import pims
import PIL
import os
import PIL.ImageOps


def save_image_at_frame(path_in, path_out, i):
    ims = pims.Bioformats(path_in)
    ims.iter_axes = "ct"
    # frame = ims[int(i)]
    # breakpoint()
    im = PIL.Image.fromarray(ims[int(i)])
    # im = im.convert('L') 
    # im = PIL.ImageOps.equalize(im, mask=None)

    im.save(path_out)
    print(f"Saving {path_out}")


save_image_at_frame(
                    snakemake.input.lif,
                    snakemake.output.png,
                    int(snakemake.wildcards.i)
                    )