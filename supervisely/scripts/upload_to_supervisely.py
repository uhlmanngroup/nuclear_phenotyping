import supervisely as sly

# address = "https://app.supervise.ly/"
# token = os.environ["API_TOKEN"]

# breakpoint()
def upload_to_supervisley(
    address, token, workspace_id, workspace_name, filename, t, z, c, input
):
    api = sly.Api(address, token)
    project = api.project.get_or_create(workspace_id=workspace_id, name=workspace_name)
    dataset = api.dataset.get_or_create(project.id, "dataset")
    # breakpoint()
    # images = glob("results/plast_cell/**/*.png", recursive=True)
    image_name = f"{filename}/t={t}_z={z}_c={c}"
    # breakpoint()
    upload_info = api.image.upload_path(dataset.id, name=image_name, path=input)
     # Construct the annotations metadata
    ann_json = {
        "description": "",
        "tags": [
            {"tag_name": "t", "value": t},
            {"tag_name": "z", "value": z},
            {"tag_name": "c", "value": c},
        ],
        # ... other metadata fields as needed ...
    }
    api.annotation.upload_json(upload_info.id, ann_json)
    # api.close()
    return image_name


def upload_to_supervisely_sm(input, output, wildcards, params):
    # breakpoint()
    return upload_to_supervisley(
        params.address,
        params.token,
        params.workspace_id,
        params.workspace_name,
        wildcards.filename,
        wildcards.t,
        wildcards.z,
        wildcards.c,
        input.png,
    )


upload_to_supervisely_sm(
    snakemake.input, snakemake.output, snakemake.wildcards, snakemake.params
)
