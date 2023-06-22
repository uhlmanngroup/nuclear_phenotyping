import supervisely as sly


def get_mask_from_image(
    address, token, workspace_id, workspace_name, dataset_name, name
):
    # address = params.address
    # token = params.token
    api = sly.Api(address, token)
    project = api.project.get_or_create(workspace_id=workspace_id, name=workspace_name)
    dataset = api.dataset.get_or_create(project.id, dataset_name)
    image_id = api.image.get_info_by_name(dataset.id, name=name).id

    # breakpoint()
    # images = glob("results/plast_cell/**/*.png", recursive=True)
    # {filename}/i={i}_t={t}_z={z}_c={c}_mask.png


def get_mask_from_image_sm(input, output, wildcards, params):
    get_mask_from_image(
        params.address,
        params.token,
        params.workspace_id,
        params.workspace_name,
        params.dataset_name,
        wildcards.name,
    )


get_mask_from_image_sm(
    snakemake.input, snakemake.output, snakemake.wildcards, snakemake.params
)
