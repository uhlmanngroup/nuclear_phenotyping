import pims
import PIL


def convert_to_pngs(path_in, dir_out):
    # breakpoint() 
    ims = pims.Bioformats(path_in)
    try:
        ims.iter_axes = "ct"
        for i, frame in enumerate(ims):
            coords = frame.metadata["coords"]
            coords["z"] = 0
            # ims = pims.Bioformats(path_in)
            # ims.iter_axes = "ct"
            # frame = ims[int(i)]
            im = PIL.Image.fromarray(ims[int(i)])
            # im = im.convert('L') 
            # im = PIL.ImageOps.equalize(im, mask=None)
            z=coords["z"]
            c=coords["c"]
            t=coords["t"]
            save_path = f"{dir_out}/i={i}_t={t}_z={z}_c={c}.png"
            # print("Saving", save_path)
            # breakpoint()
            im.save(save_path)
    except:
        print(f"Error in {path_in}")
convert_to_pngs(snakemake.input.image, snakemake.output.folder)