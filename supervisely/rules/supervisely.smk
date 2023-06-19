
import supervisely as sly

rule upload_to_supervisley:
    input:
        png="results/plast_cell/{filename}/i={i}_t={t}_z={z}_c={c}.png",
    output:
        touch("results/plast_cell/{filename}/i={i}_t={t}_z={z}_c={c}.uploaded"),
    params:
        address=config["supervisely"]["address"],
        token=config["supervisely"]["token"],
        workspace_id=config["supervisely"]["workspace_id"],
        workspace_name=config["supervisely"]["workspace_name"],
        dataset_name=config["supervisely"]["dataset_name"],
    run:

        # address = "https://app.supervise.ly/"
        # token = os.environ["API_TOKEN"]
        
        breakpoint()
        api = sly.Api(params.address, params.token)
        project = api.project.get_or_create(
            workspace_id=params.workspace_id, name=params.workspace_name
        )
        dataset = api.dataset.get_or_create(project.id, "dataset")
        # breakpoint()
        # images = glob("results/plast_cell/**/*.png", recursive=True)
        image_name = (
            f"{wildcards.filename}/t={wildcards.t}_z={wildcards.z}_c={wildcards.c}"
        )
        # breakpoint()
        api.image.upload_path(dataset.id, name=image_name, path=input.png)
        # api.close()


rule get_mask_from_image:
    input:
        "results/plast_cell/{filename}/i={i}_t={t}_z={z}_c={c}.uploaded",
    output:
        "results/plast_cell/{filename}/i={i}_t={t}_z={z}_c={c}_mask.png",
    params:
        address=config["supervisely"]["address"],
        token=config["supervisely"]["token"],
        workspace_id=config["supervisely"]["workspace_id"],
        workspace_name=config["supervisely"]["workspace_name"],
        dataset_name=config["supervisely"]["dataset_name"],
    run:
        address = params.address
        token = params.token
        api = sly.Api(address, token)
        project = api.project.get_or_create(
            workspace_id=params.workspace_id, name=params.workspace_name
        )
        dataset = api.dataset.get_or_create(project.id, params.dataset_name)
        image_id = api.image.get_info_by_name(dataset.id, name=name).id

        # breakpoint()
        # images = glob("results/plast_cell/**/*.png", recursive=True)
        # {filename}/i={i}_t={t}_z={z}_c={c}_mask.png
