import argparse
import os
from typing import Optional
import numpy as np
import torch
import torch.utils.checkpoint

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, AutoencoderKLTemporalDecoder
from transformers import CLIPTextModel, CLIPTokenizer

from backbones.video_diffusion_sd.models.unet_3d_condition import UNetPseudo3DConditionModel
from backbones.video_diffusion_sd.pipelines.stable_diffusion import SpatioTemporalStableDiffusionPipeline
from backbones.video_diffusion_sd.pnp_utils import register_spatial_attention_pnp, latent_adain

from src.util import save_folder, save_videos_grid, load_ddim_latents_at_t, seed_everything
from backbones.video_diffusion_sd.pnp_utils import map_style_alpha

STYLE_BANK = {
    "01": "results/styles-inv/sd/01/inversion",
    "02": "results/styles-inv/sd/02/inversion",
    "03": "results/styles-inv/sd/03/inversion",
    "04": "results/styles-inv/sd/04/inversion",
    "05": "results/styles-inv/sd/05/inversion",
    "06": "results/styles-inv/sd/06/inversion",
    "07": "results/styles-inv/sd/07/inversion",
    "08": "results/styles-inv/sd/08/inversion",
    "09": "results/styles-inv/sd/09/inversion",
    "10": "results/styles-inv/sd/10/inversion",
    "11": "results/styles-inv/sd/11/inversion",
    "12": "results/styles-inv/sd/12/inversion",
}

def resolve_style_path(style_path, style_id, tag):
    if style_path is not None:
        return style_path

    if style_id is None:
        raise ValueError(f"{tag}: either style_path or style_id must be provided")

    style_id = str(style_id).zfill(2)

    if style_id not in STYLE_BANK:
        raise ValueError(f"{tag}: unknown style_id={style_id}, available={list(STYLE_BANK.keys())}")

    return STYLE_BANK[style_id]

def main(
    pretrained_model_path: str,
    content_inv_path: str,
    style_inv_path: str,
    style_inv_path_b: str,
    mask_path: str,
    output_path: str,
    mapped_style_alpha: float = 0.5,
    alpha_schedule=None,
    weight_dtype: torch.dtype = torch.float16,
    #
    time_steps: int = 50,
    seed: Optional[int] = 33,
    **kwargs,
):  
    texture_styleid_enable = bool(kwargs.get("texture_styleid_enable", False))
    texture_styleid_strength = float(kwargs.get("texture_styleid_strength", 0.75))
    texture_styleid_tau = float(kwargs.get("texture_styleid_tau", 1.50))
    texture_styleid_start_step = int(kwargs.get("texture_styleid_start_step", 8))
    texture_styleid_end_step = int(kwargs.get("texture_styleid_end_step", 42))
    texture_styleid_layers = kwargs.get("texture_styleid_layers", "default")
    texture_styleid_q_strength = float(kwargs.get("texture_styleid_q_strength", 0.75))
    texture_styleid_debug = bool(kwargs.get("texture_styleid_debug", False))
    texture_styleid_dual_kv = bool(kwargs.get("texture_styleid_dual_kv", False))
    texture_styleid_init_adain_alpha = float(kwargs.get("texture_styleid_init_adain_alpha", 0.35))
    texture_styleid_init_adain_alpha = max(0.0, min(1.0, texture_styleid_init_adain_alpha))
    texture_styleid_temporal_mode = kwargs.get("texture_styleid_temporal_mode", "prev")

    #替换
    if alpha_schedule is None:
        print(f"[Main] received mapped_style_alpha={mapped_style_alpha:.4f}")
    else:
        print(f"[Main] received alpha_schedule with {len(alpha_schedule)} steps")
        print(f"[Main] alpha_schedule={alpha_schedule}")
        #print(f"[Main] received mapped_style_alpha={mapped_style_alpha:.4f}")
    if seed is not None:
        seed_everything(seed)
    # Load model
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").requires_grad_(False)
    # use 3d vae for more stable results
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "models/stable-video-diffusion-img2vid/vae",
        local_files_only=True,
    ).requires_grad_(False)
    unet = UNetPseudo3DConditionModel.from_2d_model(os.path.join(pretrained_model_path, "unet")).requires_grad_(False)
    # set device
    text_encoder = text_encoder.to(weight_dtype).cuda()
    vae = vae.to(weight_dtype).cuda()
    unet = unet.to(weight_dtype).cuda()
    # custom pipe
    pipe = SpatioTemporalStableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )

    
    content_inv_noise = load_ddim_latents_at_t(
            time_steps, ddim_latents_path=content_inv_path
        ).to(weight_dtype).cuda()
    
    style_inv_noise_a = load_ddim_latents_at_t(
            time_steps, ddim_latents_path=style_inv_path
        ).to(weight_dtype).cuda()

    style_inv_noise_b = load_ddim_latents_at_t(
            time_steps, ddim_latents_path=style_inv_path_b
        ).to(weight_dtype).cuda()

    # style_inv_noise = load_ddim_latents_at_t(
    #         time_steps, ddim_latents_path=style_inv_path
    #     ).to(weight_dtype).cuda()
    # -----------------------------------------------------------------------------------------------
    # Init latent-shift
    #替换
    if texture_styleid_dual_kv:
        print(
            f"[Dual Direct K/V Init] enabled, latent AdaIN alpha={texture_styleid_init_adain_alpha:.3f} "
            f"(set 0.0 to use pure content inversion as initial latent)"
        )
        if texture_styleid_init_adain_alpha <= 1e-6:
            inv_latents_at_t = content_inv_noise
        elif alpha_schedule is None:
            print(f"[Dual Direct K/V Init] fixed style-stat init alpha={mapped_style_alpha:.4f}")
            inv_latents_at_t = latent_adain(
                content_inv_noise,
                style_inv_noise_a,
                style_inv_noise_b,
                alpha=texture_styleid_init_adain_alpha,
                mix_alpha=mapped_style_alpha
            )
        else:
            print(f"[Dual Direct K/V Init] frame-wise style-stat init, schedule_len={len(alpha_schedule)}")
            inv_latents_at_t = latent_adain(
                content_inv_noise,
                style_inv_noise_a,
                style_inv_noise_b,
                alpha=texture_styleid_init_adain_alpha,
                mix_alpha_schedule=alpha_schedule
            )
    elif alpha_schedule is None:
        print(f"[Dual Style AdaIN] mix alpha={mapped_style_alpha:.4f}")
        inv_latents_at_t = latent_adain(
            content_inv_noise,
            style_inv_noise_a,
            style_inv_noise_b,
            alpha=1.0,
            mix_alpha=mapped_style_alpha
        )
    else:
        print(f"[Dual Style AdaIN] use alpha_schedule with {len(alpha_schedule)} steps")
        inv_latents_at_t = latent_adain(
            content_inv_noise,
            style_inv_noise_a,
            style_inv_noise_b,
            alpha=1.0,
            mix_alpha_schedule=alpha_schedule
        )
    # print(f"[Dual Style AdaIN] mix alpha={mapped_style_alpha:.4f}")
    # inv_latents_at_t = latent_adain(
    #     content_inv_noise,
    #     style_inv_noise_a,
    #     style_inv_noise_b,
    #     alpha=1.0,
    #     mix_alpha=mapped_style_alpha
    # )


    # print(f"[Latent AdaIN] use alpha={mapped_style_alpha:.4f}")
    # inv_latents_at_t = latent_adain(
    #     content_inv_noise,
    #     style_inv_noise,
    #     alpha=mapped_style_alpha
    # )


    #inv_latents_at_t = latent_adain(content_inv_noise, style_inv_noise)
    # -----------------------------------------------------------------------------------------------
    # Init Pnp, modify attention forward
    #新增
    # if alpha_schedule is None:
    #     attn_mix_alpha = mapped_style_alpha
    # else:
    #     attn_mix_alpha = float(alpha_schedule[len(alpha_schedule) // 2])
    #     print(f"[Attention PnP] schedule mode, use middle alpha={attn_mix_alpha:.4f} for attention registration")
    # register_spatial_attention_pnp(pipe, mix_alpha=attn_mix_alpha)    
    #register_spatial_attention_pnp(pipe, mix_alpha=mapped_style_alpha)
    if alpha_schedule is None:
        print(f"[Attention PnP] fixed mode, alpha={mapped_style_alpha:.4f}")
        register_spatial_attention_pnp(
            pipe,
            mix_alpha=mapped_style_alpha,
            mix_alpha_schedule=None,
            dual_style_kv=True,
            texture_styleid_enable=texture_styleid_enable,
            texture_styleid_strength=texture_styleid_strength,
            texture_styleid_tau=texture_styleid_tau,
            texture_styleid_start_step=texture_styleid_start_step,
            texture_styleid_end_step=texture_styleid_end_step,
            texture_styleid_layers=texture_styleid_layers,
            texture_styleid_q_strength=texture_styleid_q_strength,
            texture_styleid_debug=texture_styleid_debug,
            texture_styleid_dual_kv=texture_styleid_dual_kv,
            texture_styleid_temporal_mode=texture_styleid_temporal_mode,
        )
    else:
        if texture_styleid_dual_kv:
            # In V1-G, alpha_schedule is frame-wise: style-A -> style-B across video frames.
            # Do NOT resample it to denoising steps; otherwise the texture transition becomes blurred.
            attn_alpha_schedule = alpha_schedule
            attn_mix_alpha = float(alpha_schedule[len(alpha_schedule) // 2])
            print(
                f"[Attention PnP] V1-G frame-wise alpha schedule, "
                f"len={len(attn_alpha_schedule)}, start={attn_alpha_schedule[0]:.4f}, "
                f"end={attn_alpha_schedule[-1]:.4f}"
            )
        else:
            # Original V1 behavior: use the schedule only as a denoising-time routing bias.
            src_x = np.linspace(0.0, 1.0, len(alpha_schedule))
            dst_x = np.linspace(0.0, 1.0, time_steps)
            attn_alpha_schedule = np.interp(dst_x, src_x, alpha_schedule).tolist()
            attn_mix_alpha = float(attn_alpha_schedule[len(attn_alpha_schedule) // 2])

            print(f"[Attention PnP] schedule mode, resample alpha_schedule "
                f"from {len(alpha_schedule)} to {len(attn_alpha_schedule)} steps")
            print(f"[Attention PnP] attn_alpha_schedule={attn_alpha_schedule}")

        register_spatial_attention_pnp(
            pipe,
            mix_alpha=attn_mix_alpha,
            mix_alpha_schedule=attn_alpha_schedule,
            dual_style_kv=True,
            texture_styleid_enable=texture_styleid_enable,
            texture_styleid_strength=texture_styleid_strength,
            texture_styleid_tau=texture_styleid_tau,
            texture_styleid_start_step=texture_styleid_start_step,
            texture_styleid_end_step=texture_styleid_end_step,
            texture_styleid_layers=texture_styleid_layers,
            texture_styleid_q_strength=texture_styleid_q_strength,
            texture_styleid_debug=texture_styleid_debug,
            texture_styleid_dual_kv=texture_styleid_dual_kv,
            texture_styleid_temporal_mode=texture_styleid_temporal_mode,
        )


    #register_spatial_attention_pnp(pipe)
    # video_style_transfer
    #替换
    if alpha_schedule is None:
        sample = pipe.video_style_transfer(
            "",
            latents=inv_latents_at_t,
            num_inference_steps=time_steps,
            content_inv_path=content_inv_path,
            style_inv_path=style_inv_path,
            style_inv_path_b=style_inv_path_b,
            mix_alpha=mapped_style_alpha,
            mask_path=mask_path,
            texture_styleid_dual_kv=texture_styleid_dual_kv
        ).images
    else:
        sample = pipe.video_style_transfer(
            "",
            latents=inv_latents_at_t,
            num_inference_steps=time_steps,
            content_inv_path=content_inv_path,
            style_inv_path=style_inv_path,
            style_inv_path_b=style_inv_path_b,
            mix_alpha_schedule=alpha_schedule,
            mask_path=mask_path,
            texture_styleid_dual_kv=texture_styleid_dual_kv
        ).images

    # sample = pipe.video_style_transfer(
    #     "",
    #     latents=inv_latents_at_t,
    #     num_inference_steps=time_steps,
    #     content_inv_path=content_inv_path,
    #     style_inv_path=style_inv_path,
    #     style_inv_path_b=style_inv_path_b,
    #     mix_alpha=mapped_style_alpha,
    #     mask_path=mask_path
    # ).images


    # sample = pipe.video_style_transfer("", latents=inv_latents_at_t,
    #                                    num_inference_steps=time_steps,
    #                                    content_inv_path=content_inv_path, style_inv_path=style_inv_path, mask_path=mask_path).images
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    # save
    output_path = os.path.join(output_path, 'sd', f'{content_inv_path.split("/")[-2]}_{style_inv_path.split("/")[-2]}')
    os.makedirs(output_path, exist_ok=True)
    save_folder(sample, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    # parser.add_argument("--pretrained_model_path", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--content_inv_path", type=str, default="results/contents-inv/sd/mallard-fly/inversion")
    #替换
    parser.add_argument("--style_inv_path", type=str, default=None,
                    help="Path to style A inversion")

    parser.add_argument("--style_inv_path_b", type=str, default=None,
                        help="Path to style B inversion")

    parser.add_argument("--style_id_a", type=str, default="01",
                        help="Style bank ID for style A")

    parser.add_argument("--style_id_b", type=str, default="06",
                        help="Style bank ID for style B")


    # parser.add_argument("--style_inv_path", type=str, default="results/styles-inv/sd/01/inversion",
    #                     help="First style inversion path")
    # parser.add_argument("--style_inv_path_b", type=str, default="results/styles-inv/sd/02/inversion",
    #                     help="Second style inversion path")
    parser.add_argument("--mask_path", type=str, default="results/masks/sd/mallard-fly")
    parser.add_argument("--output_path", type=str, default="results/stylizations")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33)
    #新增
    parser.add_argument("--style_alpha", type=float, default=0.5,
                        help="User input style control value in [0, 1]")
    
    #替换
    parser.add_argument("--alpha_mode", type=str, default="linear", choices=["linear", "sigmoid"],
                    help="Mapping mode for style alpha")
    
    # parser.add_argument("--alpha_mode", type=str, default="sigmoid",
    #                     choices=["linear", "sigmoid"],
    #                     help="Mapping mode for style alpha")
    parser.add_argument("--alpha_beta", type=float, default=6.0,
                        help="Sigmoid sharpness for nonlinear alpha mapping")
    #再新增
    parser.add_argument("--use_alpha_schedule", action="store_true",
                    help="Use a time-varying alpha schedule instead of a single alpha")

    #替换
    parser.add_argument("--alpha_start", type=float, default=0.15,
                    help="Start alpha for schedule mode")
    # parser.add_argument("--alpha_start", type=float, default=0.0,
    #                     help="Start alpha for schedule mode")

    #替换
    parser.add_argument("--alpha_end", type=float, default=0.85,
                    help="End alpha for schedule mode")
    
    # parser.add_argument("--alpha_end", type=float, default=1.0,
    #                     help="End alpha for schedule mode")


    
    parser.add_argument("--alpha_steps", type=int, default=16,
                        help="Number of alpha steps in schedule mode")

    # StyleID-Video V1: training-free texture K/V injection.
    parser.add_argument("--texture_styleid_enable", action="store_true",
                        help="Enable StyleID-Video V1 texture injection in selected self-attention layers")
    parser.add_argument("--texture_styleid_strength", type=float, default=0.75,
                        help="Blend strength for style K/V into the edit branch, range [0,1]")
    parser.add_argument("--texture_styleid_tau", type=float, default=1.50,
                        help="Attention temperature/sharpness multiplier applied to the edit query")
    parser.add_argument("--texture_styleid_start_step", type=int, default=8,
                        help="First denoising step index to enable StyleID-Video injection")
    parser.add_argument("--texture_styleid_end_step", type=int, default=42,
                        help="Last denoising step index to enable StyleID-Video injection")
    parser.add_argument("--texture_styleid_layers", type=str, default="default",
                        help="Layer spec, e.g. 'default', 'all', or '2:0,2:1,2:2,3:0,3:1,3:2'")
    parser.add_argument("--texture_styleid_q_strength", type=float, default=0.75,
                        help="Blend strength for content Q into the edit branch, range [0,1]")
    parser.add_argument("--texture_styleid_debug", action="store_true",
                        help="Print first-hit debug info for each StyleID-Video attention layer")
    parser.add_argument("--texture_styleid_dual_kv", action="store_true",
                        help="Enable V1-G four-branch dual-style direct K/V routing: content/styleA/styleB/edit")
    parser.add_argument("--texture_styleid_init_adain_alpha", type=float, default=0.35,
                        help="Initial latent AdaIN strength used only in V1-G. Use 0.0 for pure content initialization.")
    parser.add_argument("--texture_styleid_temporal_mode", type=str, default="prev",
                        choices=["first", "prev", "local"],
                        help="Temporal routing mode for texture K/V layers: first=original UniVST, prev=previous-only, local=frame-local")

    # Deprecated/no-op compatibility arguments.
    # Alpha-only version: foreground/background injection-strength control is removed.
    parser.add_argument("--texture_mask_path", type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    #新增
    args.style_inv_path = resolve_style_path(args.style_inv_path, args.style_id_a, "style_a")
    args.style_inv_path_b = resolve_style_path(args.style_inv_path_b, args.style_id_b, "style_b")

    print(f"[Style A] id={args.style_id_a} -> path={args.style_inv_path}")
    print(f"[Style B] id={args.style_id_b} -> path={args.style_inv_path_b}")
    mapped_alpha = map_style_alpha(
        args.style_alpha,
        mode=args.alpha_mode,
        beta=args.alpha_beta
    )
    #替换
    if args.use_alpha_schedule:
        raw_schedule = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps).tolist()
        mapped_schedule = [
            map_style_alpha(x, mode=args.alpha_mode, beta=args.alpha_beta)
            for x in raw_schedule
        ]

        print(f"[Alpha Schedule] start={args.alpha_start:.4f}, "
            f"end={args.alpha_end:.4f}, steps={args.alpha_steps}, "
            f"mode={args.alpha_mode}, beta={args.alpha_beta:.2f}")

        print(f"[Alpha Schedule Raw] {raw_schedule}")
        print(f"[Alpha Schedule Mapped] {mapped_schedule}")

        args.mapped_style_alpha = None
        args.alpha_schedule = mapped_schedule
    else:
        mapped_alpha = map_style_alpha(
            args.style_alpha,
            mode=args.alpha_mode,
            beta=args.alpha_beta
        )

        print(f"[Style Alpha] input={args.style_alpha:.4f}, "
            f"mode={args.alpha_mode}, beta={args.alpha_beta:.2f}, "
            f"mapped={mapped_alpha:.4f}")

        args.mapped_style_alpha = mapped_alpha
        args.alpha_schedule = None
        print(f"[Args Check] mapped_style_alpha={args.mapped_style_alpha:.4f}")
    # print(f"[Style Alpha] input={args.style_alpha:.4f}, "
    #     f"mode={args.alpha_mode}, beta={args.alpha_beta:.2f}, "
    #     f"mapped={mapped_alpha:.4f}")
    # args.mapped_style_alpha = mapped_alpha
    # print(f"[Args Check] mapped_style_alpha={args.mapped_style_alpha:.4f}")
    args_dict = vars(args)
    #新增
    args_dict.pop("style_id_a", None)
    args_dict.pop("style_id_b", None)


    # 只把映射后的 alpha 传给 main，原始控制参数不直接传
    args_dict.pop("style_alpha", None)
    args_dict.pop("alpha_mode", None)
    args_dict.pop("alpha_beta", None)
    #再新增
    args_dict.pop("use_alpha_schedule", None)
    args_dict.pop("alpha_start", None)
    args_dict.pop("alpha_end", None)
    args_dict.pop("alpha_steps", None)

    # Remove deprecated/no-op foreground/background control args.
    args_dict.pop("texture_mask_path", None)

    main(**args_dict)
