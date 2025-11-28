import argparse
import torch # type: ignore
import time

from opensora.datasets.utils import read_from_path
from opensora.utils.config_utils import read_config
from opensora.datasets.aspect import get_image_size
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.misc import to_torch_dtype
from opensora.utils.inference_utils import prepare_multi_resolution_info
from opensora.datasets import save_sample
from opensora.utils.inference_utils import deflicker, super_resolution

def get_latent_representation(v, vae):
    # need to ensure v has length accepted by vae
    actual_t = v.size(1)
    if vae.micro_frame_size is None:
        target_t = (actual_t - 1) // 4 * 4 + 1
    elif not vae.temporal_overlap:
        target_t = actual_t // vae.micro_frame_size * vae.micro_frame_size
    else:
        target_t = (actual_t - 1) // (vae.micro_frame_size - 1) * (vae.micro_frame_size - 1) + 1
    v = v[:, :target_t]
    v_x = vae.encode(v.unsqueeze(0).to(vae.device, vae.dtype)) # v_x: [C, T, H, W]
    return v_x

def main():
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Parse Arguments === # 
    parser = argparse.ArgumentParser()

    parser.add_argument("video_path", help="Target video file path")
    parser.add_argument("--config", required=False, default="configs/opensora-v1-3/inference/t2v.py", type=str, help="Model configuration python file")
    parser.add_argument("--caption", required=False, default="", help="Caption for input video")
    parser.add_argument("--savepath", required=False, default="samples/samples/regeneration.mp4", help="Output path for regeneration from predicted noise")

    args = parser.parse_args()

    cfg = read_config(args.config)

    cfg_dtype = cfg.get("dtype", "fp32") # type: ignore
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16")) # type: ignore

    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))

    # === Read Arguments / Config === #
    v_path = args.video_path
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    
    # === Load Video === #
    print("Loading video")
    v = read_from_path(v_path, image_size, transform_name="resize_crop")
    num_frames = v.shape[1]

    # === Build VAE === #
    print("Building VAE")
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval() # type: ignore
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)

    # === Build Scheduler === #
    print("Building Diffusion Scheduler")
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # === Build Text Encoder === #
    print("Building Text Encoder")
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)

    # === Build Model === #
    print("Building Diffusion Model")
    multi_resolution = cfg.get("multi_resolution", None)
    model_args = prepare_multi_resolution_info(
        multi_resolution, 1, image_size, num_frames, fps, device, dtype
    )
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim, # type: ignore
            model_max_length=text_encoder.model_max_length, # type: ignore
        )
        .to(device, dtype) # type: ignore
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # type: ignore # HACK: for classifier-free guidance

    # === Extract Video Latent === #
    print("Extracting video latent")
    video_latent = get_latent_representation(v, vae)

    # === Invert Video Latent Into Noise === #
    print(f"Inverting video latent into initial noise latent")
    blank_prompt = [""]
    pred_init_latent = scheduler.sample( # type: ignore
        model,
        text_encoder,
        additional_args=model_args,
        z=video_latent,
        prompts=blank_prompt,
        device=device,
        reverse=True
    )

    # == Regenerating Video From Inverse Latent === #
    print("Regenerating video from predicted noise")
    video = scheduler.sample( # type: ignore
        model,
        text_encoder,
        additional_args=model_args,
        z=pred_init_latent,
        prompts=[args.caption],
        device=device
    )
    video = video.squeeze(0) # latent [C, T, H, W]

    # === Decoding Latent to Video === #
    print("Decoding latent to video")
    t_cut = video.size(1) // 5 * 5
    if t_cut < video.size(1):
        video = video[:, :t_cut]

    video = vae.decode(video.to(dtype), num_frames=t_cut * 17 // 5).squeeze(0)

    save_path = save_sample(
        video,
        fps=save_fps,
        save_path=args.savepath,
    )
    if save_path.endswith(".mp4") and cfg.get("deflicker", False): # type: ignore
        time.sleep(1)
        save_path = deflicker(save_path)
    if save_path.endswith(".mp4") and cfg.get("super_resolution", False): # type: ignore
        time.sleep(1)
        save_path = super_resolution(save_path, cfg.get("super_resolution"))

    print("Done!")

if __name__ == "__main__":
    main()