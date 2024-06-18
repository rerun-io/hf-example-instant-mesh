from __future__ import annotations

import os
import shutil
import threading
from queue import SimpleQueue
from typing import Any

import gradio as gr
import numpy as np
import rembg
import rerun as rr
import rerun.blueprint as rrb
import spaces
import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from einops import rearrange
from gradio_rerun import Rerun
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms import v2

from src.models.lrm_mesh import InstantMesh
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_circular_camera_poses,
    get_zero123plus_input_cameras,
)
from src.utils.infer_util import remove_background, resize_foreground
from src.utils.train_util import instantiate_from_config


def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    """Get the rendering camera parameters."""
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(50.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


###############################################################################
# Configuration.
###############################################################################


def find_cuda():
    # Check if CUDA_HOME or CUDA_PATH environment variables are set
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if cuda_home and os.path.exists(cuda_home):
        return cuda_home

    # Search for the nvcc executable in the system's PATH
    nvcc_path = shutil.which("nvcc")

    if nvcc_path:
        # Remove the 'bin/nvcc' part to get the CUDA installation path
        cuda_path = os.path.dirname(os.path.dirname(nvcc_path))
        return cuda_path

    return None


cuda_path = find_cuda()

if cuda_path:
    print(f"CUDA installation found at: {cuda_path}")
else:
    print("CUDA installation not found")

config_path = "configs/instant-mesh-large.yaml"
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace(".yaml", "")
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith("instant-mesh") else False

device = torch.device("cuda")

# load diffusion model
print("Loading diffusion model ...")
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

# load custom white-background UNet
unet_ckpt_path = hf_hub_download(
    repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model"
)
state_dict = torch.load(unet_ckpt_path, map_location="cpu")
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)
print(f"type(pipeline)={type(pipeline)}")

# load reconstruction model
print("Loading reconstruction model ...")
model_ckpt_path = hf_hub_download(
    repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model"
)
model: InstantMesh = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.") and "source_camera" not in k}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)

print("Loading Finished!")


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background):
    rembg_session = rembg.new_session() if do_remove_background else None

    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    return input_image


def pipeline_callback(
    log_queue: SimpleQueue, pipe: Any, step_index: int, timestep: float, callback_kwargs: dict[str, Any]
) -> dict[str, Any]:
    latents = callback_kwargs["latents"]
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]  # type: ignore[attr-defined]
    image = pipe.image_processor.postprocess(image, output_type="np").squeeze()  # type: ignore[attr-defined]

    log_queue.put(("mvs", rr.Image(image)))
    log_queue.put(("latents", rr.Tensor(latents.squeeze())))

    return callback_kwargs


def generate_mvs(log_queue, input_image, sample_steps, sample_seed):
    seed_everything(sample_seed)

    return pipeline(
        input_image,
        num_inference_steps=sample_steps,
        callback_on_step_end=lambda *args, **kwargs: pipeline_callback(log_queue, *args, **kwargs),
    ).images[0]


def make3d(log_queue, images: Image.Image):
    global model
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, use_renderer=False)
    model = model.eval()

    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
    images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)  # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)

    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get mesh
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out

        log_queue.put((
            "mesh",
            rr.Mesh3D(vertex_positions=vertices, vertex_colors=vertex_colors, triangle_indices=faces),
        ))

    return mesh_out


def generate_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="mesh"),
            rrb.Grid(
                rrb.Spatial2DView(origin="z123image"),
                rrb.Spatial2DView(origin="preprocessed_image"),
                rrb.Spatial2DView(origin="mvs"),
                rrb.TensorView(
                    origin="latents",
                ),
            ),
            column_shares=[1, 1],
        ),
        collapse_panels=True,
    )


def compute(log_queue, input_image, do_remove_background, sample_steps, sample_seed):
    preprocessed_image = preprocess(input_image, do_remove_background)
    log_queue.put(("preprocessed_image", rr.Image(preprocessed_image)))

    z123_image = generate_mvs(log_queue, preprocessed_image, sample_steps, sample_seed)
    log_queue.put(("z123image", rr.Image(z123_image)))

    _mesh_out = make3d(log_queue, z123_image)

    log_queue.put("done")


@spaces.GPU
@rr.thread_local_stream("InstantMesh")
def log_to_rr(input_image, do_remove_background, sample_steps, sample_seed):
    log_queue = SimpleQueue()

    stream = rr.binary_stream()

    blueprint = generate_blueprint()
    rr.send_blueprint(blueprint)
    yield stream.read()

    handle = threading.Thread(
        target=compute, args=[log_queue, input_image, do_remove_background, sample_steps, sample_seed]
    )
    handle.start()
    while True:
        msg = log_queue.get()
        if msg == "done":
            break
        else:
            entity_path, entity = msg
            rr.log(entity_path, entity)
            yield stream.read()
    handle.join()


_HEADER_ = """
<h2><b>Duplicate of the <a href='https://huggingface.co/spaces/TencentARC/InstantMesh'>InstantMesh space</a> that uses <a href='https://rerun.io/'>Rerun</a> for visualization.</b></h2>
<h2><a href='https://github.com/TencentARC/InstantMesh' target='_blank'><b>InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models</b></a></h2>

**InstantMesh** is a feed-forward framework for efficient 3D mesh generation from a single image based on the LRM/Instant3D architecture.

Technical report: <a href='https://arxiv.org/abs/2404.07191' target='_blank'>ArXiv</a>.
Source code: <a href='https://github.com/rerun-io/hf-example-instant-mesh'>Github</a>.
"""

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    # width=256,
                    # height=256,
                    type="pil",
                    elem_id="content_image",
                )
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(label="Remove Background", value=True)
                    sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

                    sample_steps = gr.Slider(label="Sample Steps", minimum=30, maximum=75, value=75, step=5)

            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")

            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))],
                    inputs=[input_image],
                    label="Examples",
                    cache_examples=False,
                    examples_per_page=16,
                )

        with gr.Column(scale=2):
            viewer = Rerun(streaming=True, height=800)

            with gr.Row():
                gr.Markdown("""Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).""")

    mv_images = gr.State()

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=log_to_rr, inputs=[input_image, do_remove_background, sample_steps, sample_seed], outputs=[viewer]
    )

demo.launch()
