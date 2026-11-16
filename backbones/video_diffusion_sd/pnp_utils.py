import torch.nn.functional as F
import torch
from einops import rearrange

import math



def map_style_alpha(s, mode="sigmoid", beta=6.0):
    """
    将用户输入的风格控制值 s ∈ [0,1]
    映射成非线性的 alpha，避免简单线性插值。

    Args:
        s: float, 用户输入，理论范围 [0, 1]
        mode: "linear" or "sigmoid"
        beta: sigmoid 曲线陡峭程度，推荐先用 6.0

    Returns:
        alpha: float, 映射后的插值系数，范围 [0, 1]
    """
    s = float(s)
    s = max(0.0, min(1.0, s))

    if mode == "linear":
        return s

    if mode == "sigmoid":
        def sig(x):
            return 1.0 / (1.0 + math.exp(-x))

        lo = sig(-0.5 * beta)
        hi = sig(0.5 * beta)
        alpha = (sig(beta * (s - 0.5)) - lo) / (hi - lo)

        # 再保险裁剪一次
        alpha = max(0.0, min(1.0, alpha))
        return alpha

    raise ValueError(f"Unsupported alpha mode: {mode}")




# Modified from tokenflow_utils.py
def register_time(model, t):
    up_res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}

    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, "idx", t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, "idx", t)

def _parse_styleid_layer_spec(layer_spec):
    """
    Parse layer spec for StyleID-Video texture injection.

    Accepted forms:
      - None / "" / "default": use high-resolution texture layers.
      - "all": use all registered decoder self-attention layers.
      - "2:0,2:1,2:2,3:0,3:1,3:2"
      - "up_blocks.2.attentions.0,up_blocks.3.attentions.1"
    Returns a set of (res, block) tuples.
    """
    default_layers = {(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)}
    all_layers = {(1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)}

    if layer_spec is None:
        return default_layers

    if isinstance(layer_spec, (list, tuple, set)):
        parsed = set()
        for item in layer_spec:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                parsed.add((int(item[0]), int(item[1])))
        return parsed if parsed else default_layers

    spec = str(layer_spec).strip()
    if spec == "" or spec.lower() == "default":
        return default_layers
    if spec.lower() == "all":
        return all_layers

    parsed = set()
    for raw_token in spec.split(','):
        token = raw_token.strip()
        if not token:
            continue

        # Short form: "2:1"
        if ':' in token and 'up_blocks' not in token:
            res_str, block_str = token.split(':', 1)
            parsed.add((int(res_str), int(block_str)))
            continue

        # Long form: "up_blocks.2.attentions.1"
        parts = token.split('.')
        if 'up_blocks' in parts and 'attentions' in parts:
            res = int(parts[parts.index('up_blocks') + 1])
            block = int(parts[parts.index('attentions') + 1])
            parsed.add((res, block))
            continue

        raise ValueError(
            f"Unsupported texture_styleid_layers token: {token}. "
            f"Use forms like '2:0,2:1,3:0' or 'all'."
        )

    return parsed if parsed else default_layers



def register_spatial_attention_pnp(
    model,
    eta1=0.0,
    eta2=0.5,
    mix_alpha=0.5,
    mix_alpha_schedule=None,
    dual_style_kv=False,
    texture_styleid_enable=False,
    texture_styleid_strength=0.75,
    texture_styleid_tau=1.50,
    texture_styleid_start_step=8,
    texture_styleid_end_step=42,
    texture_styleid_layers="default",
    texture_styleid_q_strength=0.75,
    texture_styleid_debug=False,
    texture_styleid_dual_kv=False,
    texture_styleid_frame_local=True,
    texture_styleid_temporal_mode="prev",
):
    """
    Register self-attention PnP for UniVST.

    Baseline mode keeps the original AdaIN-guided attention shift.
    StyleID-Video mode is enabled by texture_styleid_enable=True and performs
    a training-free texture injection on selected decoder self-attention layers:

        Q_edit <- blend(Q_edit, Q_content)
        K_edit <- blend(K_edit, K_style)
        V_edit <- blend(V_edit, V_style)

    This uses the existing three-way batch layout in the pipeline:
        [content inversion, style inversion, edited latents]
    """
    mix_alpha = float(max(0.0, min(1.0, mix_alpha)))
    texture_styleid_enable = bool(texture_styleid_enable)
    texture_styleid_strength = float(max(0.0, min(1.0, texture_styleid_strength)))
    texture_styleid_tau = float(max(0.5, texture_styleid_tau))
    texture_styleid_start_step = int(texture_styleid_start_step)
    texture_styleid_end_step = int(texture_styleid_end_step)
    if texture_styleid_end_step < texture_styleid_start_step:
        texture_styleid_start_step, texture_styleid_end_step = texture_styleid_end_step, texture_styleid_start_step
    texture_styleid_q_strength = float(max(0.0, min(1.0, texture_styleid_q_strength)))
    texture_styleid_dual_kv = bool(texture_styleid_dual_kv)
    # V1-G3: temporal mode for texture K/V routing layers.
    #   "first" or "default": keep UniVST original [-1, "first"] temporal anchor.
    #   "local": no cross-frame K/V aggregation in texture routing layers.
    #   "prev": only use previous-frame K/V aggregation in texture routing layers.
    #           This removes first-frame style lock while retaining adjacent-frame smoothness.
    texture_styleid_frame_local = bool(texture_styleid_frame_local)
    texture_styleid_temporal_mode = str(texture_styleid_temporal_mode).lower().strip()
    if texture_styleid_temporal_mode in ["previous", "prev_only", "previous_only"]:
        texture_styleid_temporal_mode = "prev"
    if texture_styleid_temporal_mode in ["none", "self", "frame_local"]:
        texture_styleid_temporal_mode = "local"
    if texture_styleid_temporal_mode in ["orig", "original", "default", "first_frame"]:
        texture_styleid_temporal_mode = "first"
    if texture_styleid_temporal_mode not in ["local", "prev", "first"]:
        raise ValueError(
            f"Unsupported texture_styleid_temporal_mode={texture_styleid_temporal_mode}. "
            "Use 'local', 'prev', or 'first'."
        )

    styleid_layers = _parse_styleid_layer_spec(texture_styleid_layers)

    # In V1-G mode, mix_alpha_schedule is interpreted as a frame-wise style routing schedule.
    # In the original three-branch mode, it is only used to estimate a global strength bias.
    texture_styleid_frame_alpha_schedule = None
    if texture_styleid_dual_kv and mix_alpha_schedule is not None:
        texture_styleid_frame_alpha_schedule = [
            float(max(0.0, min(1.0, x))) for x in mix_alpha_schedule
        ]

    if mix_alpha_schedule is not None:
        mix_alpha_schedule = [
            float(max(0.0, min(1.0, x))) for x in mix_alpha_schedule
        ]
        mix_alpha_for_strength = sum(mix_alpha_schedule) / max(len(mix_alpha_schedule), 1)
    else:
        mix_alpha_for_strength = mix_alpha

    # Baseline layer-wise PnP configuration.
    # res=1: lower resolution, controls coarse/global style.
    # res=2: middle resolution, controls main style appearance.
    # res=3: higher resolution, most related to local texture but also more fragile.
    layer_cfg = {
        1: dict(q_alpha=0.24, kv_beta_max=0.62, kv_beta_min=0.34, gamma=2.10),
        2: dict(q_alpha=0.34, kv_beta_max=0.46, kv_beta_min=0.24, gamma=1.65),
        3: dict(q_alpha=0.58, kv_beta_max=0.16, kv_beta_min=0.06, gamma=1.08),
    }

    block_scale = {
        0: 1.00,
        1: 0.92,
        2: 0.84,
    }

    def ca_forward(self):
        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            clip_length: int = None,
            SparseCausalAttention_index: list = [-1, 'first']
        ) -> torch.FloatTensor:
            input_ndim = hidden_states.ndim

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            # UniVST V1 uses a three-way batch:
            #   [content inversion, mixed style inversion, edited latents].
            # V1-G uses a four-way batch:
            #   [content inversion, style-A inversion, style-B inversion, edited latents].
            # We only activate the four-way parser when the dual-KV flag is explicitly enabled.
            styleid_dual_kv = bool(getattr(self, "texture_styleid_dual_kv", False))
            use_four_way_pnp = styleid_dual_kv and (batch_size % 4 == 0)
            use_three_way_pnp = (not use_four_way_pnp) and (batch_size % 3 == 0)
            if use_four_way_pnp:
                chunk_size = batch_size // 4
            elif use_three_way_pnp:
                chunk_size = batch_size // 3
            else:
                chunk_size = 0

            query = self.to_q(hidden_states)
            dim = query.shape[-1]
            head_dim = dim // self.heads
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            idx = int(getattr(self, "idx", -1))
            styleid_enabled = bool(getattr(self, "texture_styleid_enable", False))
            styleid_selected = bool(getattr(self, "texture_styleid_selected", False))
            styleid_in_window = (
                idx >= int(getattr(self, "texture_styleid_start_step", 8))
                and idx <= int(getattr(self, "texture_styleid_end_step", 42))
            )
            # V1-G3: only selected texture-routing layers use the modified temporal mode.
            # Other layers keep the original UniVST temporal attention.
            styleid_temporal_override_now = (
                use_four_way_pnp
                and styleid_enabled
                and styleid_selected
                and styleid_in_window
                and bool(getattr(self, "texture_styleid_frame_local", True))
            )
            styleid_temporal_mode_now = str(getattr(self, "texture_styleid_temporal_mode", "prev")).lower().strip()

            # -------------------------------------StyleID-Video V1-G3 dual-style direct K/V routing----------
            if use_four_way_pnp and styleid_enabled and styleid_selected and styleid_in_window:
                strength = float(getattr(self, "texture_styleid_strength", 0.75))
                tau = float(getattr(self, "texture_styleid_tau", 1.50))
                q_strength = float(getattr(self, "texture_styleid_q_strength", 0.75))

                q_content = query[:chunk_size]
                q_edit = query[3 * chunk_size: 4 * chunk_size]

                k_style_a = key[chunk_size: 2 * chunk_size]
                v_style_a = value[chunk_size: 2 * chunk_size]
                k_style_b = key[2 * chunk_size: 3 * chunk_size]
                v_style_b = value[2 * chunk_size: 3 * chunk_size]

                k_edit = key[3 * chunk_size: 4 * chunk_size]
                v_edit = value[3 * chunk_size: 4 * chunk_size]

                frame_schedule = getattr(self, "texture_styleid_frame_alpha_schedule", None)
                if frame_schedule is not None and len(frame_schedule) > 0:
                    alpha = torch.tensor(frame_schedule, device=query.device, dtype=query.dtype)
                    # Usually chunk_size equals the number of frames. Interpolate defensively if needed.
                    if alpha.numel() != chunk_size:
                        alpha = F.interpolate(
                            alpha.view(1, 1, -1),
                            size=chunk_size,
                            mode="linear",
                            align_corners=True,
                        ).view(-1)
                    alpha = alpha.clamp(0.0, 1.0).view(chunk_size, 1, 1)
                else:
                    alpha_value = float(getattr(self, "texture_styleid_mix_alpha", 0.5))
                    alpha = torch.full(
                        (chunk_size, 1, 1),
                        alpha_value,
                        device=query.device,
                        dtype=query.dtype,
                    )

                # Mix style-A and style-B directly in the attention K/V space.
                # This avoids first averaging the two styles in latent/statistics space.
                k_style = (1.0 - alpha) * k_style_a + alpha * k_style_b
                v_style = (1.0 - alpha) * v_style_a + alpha * v_style_b

                # Alpha-only version: use one global fixed texture strength.
                # No foreground/background spatial strength control.
                local_strength = strength

                query[3 * chunk_size: 4 * chunk_size] = (
                    q_strength * q_content + (1.0 - q_strength) * q_edit
                )
                key[3 * chunk_size: 4 * chunk_size] = (
                    local_strength * k_style + (1.0 - local_strength) * k_edit
                )
                value[3 * chunk_size: 4 * chunk_size] = (
                    local_strength * v_style + (1.0 - local_strength) * v_edit
                )
                query[3 * chunk_size: 4 * chunk_size] = tau * query[3 * chunk_size: 4 * chunk_size]

                if bool(getattr(self, "texture_styleid_debug", False)) and not bool(getattr(self, "texture_styleid_printed", False)):
                    print(
                        f"[StyleID-Video V1-G3] layer={getattr(self, 'texture_styleid_layer_name', 'unknown')}, "
                        f"step={idx}, strength={strength:.3f}, tau={tau:.3f}, q_strength={q_strength:.3f}, "
                        f"alpha_range=[{float(alpha.min()):.3f}, {float(alpha.max()):.3f}], temporal_mode={styleid_temporal_mode_now}"
                    )
                    setattr(self, "texture_styleid_printed", True)

            # -------------------------------------StyleID-Video V1 single/mixed-style texture injection------
            elif use_three_way_pnp and styleid_enabled and styleid_selected and styleid_in_window:
                strength = float(getattr(self, "texture_styleid_strength", 0.75))
                tau = float(getattr(self, "texture_styleid_tau", 1.50))
                q_strength = float(getattr(self, "texture_styleid_q_strength", 0.75))

                q_content = query[:chunk_size]
                q_edit = query[2 * chunk_size: 3 * chunk_size]
                k_style = key[chunk_size: 2 * chunk_size]
                v_style = value[chunk_size: 2 * chunk_size]
                k_edit = key[2 * chunk_size: 3 * chunk_size]
                v_edit = value[2 * chunk_size: 3 * chunk_size]

                # Alpha-only version: use one global fixed texture strength.
                # No foreground/background spatial strength control.
                local_strength = strength

                query[2 * chunk_size: 3 * chunk_size] = (
                    q_strength * q_content + (1.0 - q_strength) * q_edit
                )
                key[2 * chunk_size: 3 * chunk_size] = (
                    local_strength * k_style + (1.0 - local_strength) * k_edit
                )
                value[2 * chunk_size: 3 * chunk_size] = (
                    local_strength * v_style + (1.0 - local_strength) * v_edit
                )
                query[2 * chunk_size: 3 * chunk_size] = tau * query[2 * chunk_size: 3 * chunk_size]

                if bool(getattr(self, "texture_styleid_debug", False)) and not bool(getattr(self, "texture_styleid_printed", False)):
                    print(
                        f"[StyleID-Video V1] layer={getattr(self, 'texture_styleid_layer_name', 'unknown')}, "
                        f"step={idx}, strength={strength:.3f}, tau={tau:.3f}, q_strength={q_strength:.3f}"
                    )
                    setattr(self, "texture_styleid_printed", True)

            # -------------------------------------Baseline AdaIN-guided attention shift----------------------
            elif use_three_way_pnp and (not styleid_enabled) and self.idx >= self.eta1 and self.idx <= self.eta2 * 50:
                q_alpha_local = getattr(self, "layer_q_alpha", 0.35)
                kv_beta_max_local = getattr(self, "layer_kv_beta_max", 0.60)
                kv_beta_min_local = getattr(self, "layer_kv_beta_min", 0.30)
                gamma_local = getattr(self, "layer_gamma", 2.0)

                # Let beta decay across denoising: stronger early, milder later.
                denom = max(1e-6, (self.eta2 - self.eta1) * 50)
                progress = (self.idx - self.eta1 * 50) / denom
                progress = float(max(0.0, min(1.0, progress)))
                beta = kv_beta_max_local + (kv_beta_min_local - kv_beta_max_local) * progress

                # Slightly stronger at alpha endpoints; more conservative in the middle.
                edge_boost = 0.95 + 0.05 * abs(2.0 * mix_alpha_for_strength - 1.0)
                beta = float(max(kv_beta_min_local, min(kv_beta_max_local, beta * edge_boost)))
                gamma_local = 1.0 + (gamma_local - 1.0) * edge_boost

                q_content = query[:chunk_size]
                q_edit = query[2 * chunk_size: 3 * chunk_size]

                k_style = key[chunk_size: 2 * chunk_size]
                v_style = value[chunk_size: 2 * chunk_size]

                k_edit = key[2 * chunk_size: 3 * chunk_size]
                v_edit = value[2 * chunk_size: 3 * chunk_size]

                query[2 * chunk_size: 3 * chunk_size] = (
                    q_alpha_local * q_content + (1.0 - q_alpha_local) * q_edit
                )
                key[2 * chunk_size: 3 * chunk_size] = (
                    beta * attention_adain(k_edit, k_style) + (1.0 - beta) * k_edit
                )
                value[2 * chunk_size: 3 * chunk_size] = (
                    beta * attention_adain(v_edit, v_style) + (1.0 - beta) * v_edit
                )
                query[2 * chunk_size: 3 * chunk_size] = gamma_local * query[2 * chunk_size: 3 * chunk_size]

            # -----------------------------------cross-frame attention--------------------------------------
            if clip_length is not None:
                key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
                value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)

                # V1-G3: modify temporal K/V aggregation only inside selected
                # texture-routing layers.
                #   local: each frame uses its own texture K/V only.
                #   prev : each frame uses previous-frame texture K/V only, avoiding
                #          first-frame style lock while improving adjacent-frame smoothness.
                #   first: keep UniVST original [-1, "first"] behavior.
                if styleid_temporal_override_now:
                    if styleid_temporal_mode_now == "local":
                        effective_sparse_index = []
                    elif styleid_temporal_mode_now == "prev":
                        effective_sparse_index = [-1]
                    else:
                        effective_sparse_index = SparseCausalAttention_index
                else:
                    effective_sparse_index = SparseCausalAttention_index

                frame_index_list = []
                if len(effective_sparse_index) > 0:
                    for index in effective_sparse_index:
                        if isinstance(index, str):
                            if index == 'first':
                                frame_index = [0] * clip_length
                            if index == 'last':
                                frame_index = [clip_length - 1] * clip_length
                            if (index == 'mid') or (index == 'middle'):
                                frame_index = [int(clip_length - 1) // 2] * clip_length
                        else:
                            assert isinstance(index, int), 'relative index must be int'
                            frame_index = torch.arange(clip_length) + index
                            frame_index = frame_index.clip(0, clip_length - 1)

                        frame_index_list.append(frame_index)

                    key = torch.cat([key[:, frame_index] for frame_index in frame_index_list], dim=2)
                    value = torch.cat([value[:, frame_index] for frame_index in frame_index_list], dim=2)

                key = rearrange(key, "b f d c -> (b f) d c", f=clip_length)
                value = rearrange(value, "b f d c -> (b f) d c", f=clip_length)

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}

    if texture_styleid_enable:
        tag = "StyleID-Video V1-G3" if texture_styleid_dual_kv else "StyleID-Video V1"
        schedule_info = ""
        if texture_styleid_dual_kv:
            if texture_styleid_frame_alpha_schedule is not None:
                schedule_info = (
                    f", dual_direct_kv=True, temporal_mode={texture_styleid_temporal_mode}, frame_alpha=[{texture_styleid_frame_alpha_schedule[0]:.3f}, "
                    f"{texture_styleid_frame_alpha_schedule[-1]:.3f}], frames={len(texture_styleid_frame_alpha_schedule)}"
                )
            else:
                schedule_info = f", dual_direct_kv=True, temporal_mode={texture_styleid_temporal_mode}, fixed_alpha={mix_alpha:.3f}"
        print(
            f"[{tag}] enable=True, strength={texture_styleid_strength:.3f}, "
            f"tau={texture_styleid_tau:.3f}, q_strength={texture_styleid_q_strength:.3f}, "
            f"step_range=[{texture_styleid_start_step}, {texture_styleid_end_step}], "
            f"layers={sorted(styleid_layers)}" + schedule_info
        )

    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1

            cfg = layer_cfg[res]
            scale = block_scale.get(block, 1.0)
            selected = (res, block) in styleid_layers

            setattr(module, "eta1", eta1)
            setattr(module, "eta2", eta2)

            setattr(module, "layer_q_alpha", cfg["q_alpha"])
            setattr(module, "layer_kv_beta_max", cfg["kv_beta_max"] * scale)
            setattr(module, "layer_kv_beta_min", cfg["kv_beta_min"] * scale)
            setattr(module, "layer_gamma", 1.0 + (cfg["gamma"] - 1.0) * scale)

            setattr(module, "texture_styleid_enable", texture_styleid_enable)
            setattr(module, "texture_styleid_selected", selected)
            setattr(module, "texture_styleid_strength", texture_styleid_strength)
            setattr(module, "texture_styleid_tau", texture_styleid_tau)
            setattr(module, "texture_styleid_start_step", texture_styleid_start_step)
            setattr(module, "texture_styleid_end_step", texture_styleid_end_step)
            setattr(module, "texture_styleid_q_strength", texture_styleid_q_strength)
            setattr(module, "texture_styleid_debug", bool(texture_styleid_debug))
            setattr(module, "texture_styleid_dual_kv", texture_styleid_dual_kv)
            setattr(module, "texture_styleid_frame_local", texture_styleid_frame_local)
            setattr(module, "texture_styleid_temporal_mode", texture_styleid_temporal_mode)
            setattr(module, "texture_styleid_mix_alpha", mix_alpha)
            setattr(module, "texture_styleid_frame_alpha_schedule", texture_styleid_frame_alpha_schedule)
            setattr(module, "texture_styleid_printed", False)
            setattr(module, "texture_styleid_layer_name", f"up_blocks.{res}.attentions.{block}.attn1")

            model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1.forward = ca_forward(module)



# def register_spatial_attention_pnp( model, eta1=0.0, eta2=0.5, mix_alpha=0.5, q_alpha=0.35, kv_beta_min=0.35, kv_beta_max=1.00, gamma=4.0):
# #def register_spatial_attention_pnp(model, eta1=0.0, eta2=0.5):
#     mix_alpha = float(max(0.0, min(1.0, mix_alpha)))
#     def ca_forward(self):
#         def forward(
#             hidden_states,
#             encoder_hidden_states=None,
#             attention_mask=None,
#             clip_length: int = None,
#             SparseCausalAttention_index: list = [-1, 'first']
#         ) -> torch.FloatTensor:
#             input_ndim = hidden_states.ndim

#             if self.group_norm is not None:
#                 hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
#             if input_ndim == 4:
#                     batch_size, channel, height, width = hidden_states.shape
#                     hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
#             batch_size, sequence_length, _ = (
#                 hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#             )
#             chunk_size = batch_size // 3

#             query = self.to_q(hidden_states)
#             dim = query.shape[-1]
#             head_dim = dim // self.heads
#             key = self.to_k(hidden_states)
#             value = self.to_v(hidden_states)


#             # -------------------------------------AdaIN-Guided Attention-shift-----------------------------
#             if self.idx >= self.eta1 and self.idx <= self.eta2 * 50:
#                # schedule beta from strong -> mild across denoising
#                 beta = (kv_beta_max - kv_beta_min) / (self.eta1 * 50 - self.eta2 * 50) * (self.idx - self.eta2 * 50) + kv_beta_min
#                 beta = float(max(kv_beta_min, min(kv_beta_max, beta)))

#                 q_content = query[:chunk_size]
#                 q_edit = query[2 * chunk_size: 3 * chunk_size]

#                 k_style = key[chunk_size: 2 * chunk_size]
#                 v_style = value[chunk_size: 2 * chunk_size]

#                 k_edit = key[2 * chunk_size: 3 * chunk_size]
#                 v_edit = value[2 * chunk_size: 3 * chunk_size]

#                 # query: keep structure, but reduce content dominance a bit
#                 query[2 * chunk_size: 3 * chunk_size] = q_alpha * q_content + (1.0 - q_alpha) * q_edit

#                 # key/value: use mixed-style chunk to push texture into edit branch
#                 key[2 * chunk_size: 3 * chunk_size] = beta * attention_adain(k_edit, k_style) + (1.0 - beta) * k_edit
#                 value[2 * chunk_size: 3 * chunk_size] = beta * attention_adain(v_edit, v_style) + (1.0 - beta) * v_edit

#                 # stronger attention response on edited branch
#                 query[2 * chunk_size: 3 * chunk_size] = gamma * query[2 * chunk_size: 3 * chunk_size]
#             # if self.idx >= self.eta1 and self.idx <= self.eta2 * 50:
#             #     # parameter
#             #     alpha = 0.65
#             #     beta = (0.9 - 0.1) / (self.eta1 * 50 - self.eta2 * 50) * (self.idx - self.eta2 * 50) + 0.1
#             #     gamma = 3.0
#             #     #
#             #     query[2 * chunk_size: 3 * chunk_size] = alpha * query[: chunk_size] + (1 - alpha) * query[2 * chunk_size: 3 * chunk_size]  
#             #     key[2 * chunk_size: 3 * chunk_size] = beta * attention_adain(key[2 * chunk_size: 3 * chunk_size], key[chunk_size: 2 * chunk_size]) + (1 - beta) * key[chunk_size: 2 * chunk_size]
#             #     value[2 * chunk_size: 3 * chunk_size] = beta * attention_adain(value[2 * chunk_size: 3 * chunk_size], value[chunk_size: 2 * chunk_size]) + (1 - beta) * value[chunk_size: 2 * chunk_size]
#             #     # attn argue
#             #     query[2 * chunk_size: 3 * chunk_size] = gamma * query[2 * chunk_size: 3 * chunk_size]
#             # -----------------------------------cross-frame attention--------------------------------------
#             if clip_length is not None:
#                 key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
#                 value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)
#                 #  *********************** Start of Spatial-temporal attention **********
#                 frame_index_list = []
#                 if len(SparseCausalAttention_index) > 0:
#                     for index in effective_sparse_index:
#                         if isinstance(index, str):
#                             if index == 'first':
#                                 frame_index = [0] * clip_length
#                             if index == 'last':
#                                 frame_index = [clip_length - 1] * clip_length
#                             if (index == 'mid') or (index == 'middle'):
#                                 frame_index = [int(clip_length-1)//2] * clip_length
#                         else:
#                             assert isinstance(index, int), 'relative index must be int'
#                             frame_index = torch.arange(clip_length) + index
#                             frame_index = frame_index.clip(0, clip_length-1)        
#                         frame_index_list.append(frame_index) 
#                     key = torch.cat([   key[:, frame_index] for frame_index in frame_index_list
#                                         ], dim=2)
#                     value = torch.cat([ value[:, frame_index] for frame_index in frame_index_list
#                                         ], dim=2)
#                 #  *********************** End of Spatial-temporal attention **********
#                 key = rearrange(key, "b f d c -> (b f) d c", f=clip_length)
#                 value = rearrange(value, "b f d c -> (b f) d c", f=clip_length)
#             # ----------------------------------------------------------------------------------------
#             query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#             key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#             value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

#             hidden_states = F.scaled_dot_product_attention(
#                 query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#             )

#             hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
#             hidden_states = hidden_states.to(query.dtype)
#             # linear proj
#             hidden_states = self.to_out[0](hidden_states)
#             # dropout
#             hidden_states = self.to_out[1](hidden_states)
#             return hidden_states
#         return forward


#     res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
#     # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
#     for res in res_dict:
#         for block in res_dict[res]:
#             module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
#             setattr(module, "eta1", eta1)
#             setattr(module, "eta2", eta2)
#             model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1.forward = ca_forward(module)


def attention_adain(cnt_feat, sty_feat, ad=True):
    beta = 1.0
    cnt_mean = cnt_feat.mean(dim=[1],keepdim=True)
    cnt_std = cnt_feat.std(dim=[1],keepdim=True)
    sty_mean = sty_feat.mean(dim=[1],keepdim=True)
    sty_std = sty_feat.std(dim=[1],keepdim=True)
    output_mean = beta * sty_mean + (1 - beta) * cnt_mean
    output_std = beta * sty_std + (1 - beta) * cnt_std

    
    output_mean = beta * sty_mean + (1 - beta) * cnt_mean
    output_std = beta * sty_std + (1 - beta) * cnt_std
    if ad:
        output = F.instance_norm(cnt_feat) * output_std + output_mean

    return output.to(cnt_feat.dtype)


# def latent_adain(cnt_feat, sty_feat, ad=True, alpha=1.0):
#def latent_adain(cnt_feat, sty_feat_a, sty_feat_b=None, ad=True, alpha=1.0, mix_alpha=0.5):
def latent_adain(
    cnt_feat,
    sty_feat_a,
    sty_feat_b=None,
    ad=True,
    alpha=1.0,
    mix_alpha=0.5,
    mix_alpha_schedule=None,
):
    beta = 1.0
    cnt_mean = cnt_feat.mean(dim=[0, 3, 4], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 3, 4], keepdim=True)
    #替换
    sty_mean_a = sty_feat_a.mean(dim=[0, 3, 4], keepdim=True)
    sty_std_a = sty_feat_a.std(dim=[0, 3, 4], keepdim=True) + 1e-6

    if sty_feat_b is not None:
        sty_mean_b = sty_feat_b.mean(dim=[0, 3, 4], keepdim=True)
        sty_std_b = sty_feat_b.std(dim=[0, 3, 4], keepdim=True) + 1e-6

        if mix_alpha_schedule is not None:
            num_frames = sty_feat_a.shape[2]
            if len(mix_alpha_schedule) != num_frames:
                raise ValueError(
                    f"mix_alpha_schedule length ({len(mix_alpha_schedule)}) "
                    f"must match number of frames ({num_frames})"
                )

            mix_alpha_tensor = torch.tensor(
                mix_alpha_schedule,
                device=sty_feat_a.device,
                dtype=sty_feat_a.dtype
            ).view(1, 1, num_frames, 1, 1)

            mix_alpha_tensor = mix_alpha_tensor.clamp(0.0, 1.0)

            sty_mean = (1.0 - mix_alpha_tensor) * sty_mean_a + mix_alpha_tensor * sty_mean_b
            sty_std = (1.0 - mix_alpha_tensor) * sty_std_a + mix_alpha_tensor * sty_std_b
        else:
            mix_alpha = float(max(0.0, min(1.0, mix_alpha)))
            sty_mean = (1.0 - mix_alpha) * sty_mean_a + mix_alpha * sty_mean_b
            sty_std = (1.0 - mix_alpha) * sty_std_a + mix_alpha * sty_std_b
    else:
        sty_mean = sty_mean_a
        sty_std = sty_std_a
    # sty_mean_a = sty_feat_a.mean(dim=[0, 3, 4], keepdim=True)
    # sty_std_a = sty_feat_a.std(dim=[0, 3, 4], keepdim=True) + 1e-6

    # if sty_feat_b is not None:
    #     sty_mean_b = sty_feat_b.mean(dim=[0, 3, 4], keepdim=True)
    #     sty_std_b = sty_feat_b.std(dim=[0, 3, 4], keepdim=True) + 1e-6

    #     mix_alpha = float(max(0.0, min(1.0, mix_alpha)))
    #     sty_mean = (1.0 - mix_alpha) * sty_mean_a + mix_alpha * sty_mean_b
    #     sty_std = (1.0 - mix_alpha) * sty_std_a + mix_alpha * sty_std_b
    # else:
    #     sty_mean = sty_mean_a
    #     sty_std = sty_std_a



    # sty_mean = sty_feat.mean(dim=[0, 3, 4], keepdim=True)
    # sty_std = sty_feat.std(dim=[0, 3, 4], keepdim=True)
    output_mean = beta * sty_mean + (1 - beta) * cnt_mean
    output_std = beta * sty_std + (1 - beta) * cnt_std
    if ad:
        styled_feat = F.instance_norm(cnt_feat) * output_std + output_mean
    else:
        styled_feat = cnt_feat

    alpha = float(max(0.0, min(1.0, alpha)))
    output = (1.0 - alpha) * cnt_feat + alpha * styled_feat
    # if ad:
    #     output = F.instance_norm(cnt_feat) * output_std + output_mean

    return output.to(cnt_feat.dtype)


if __name__ == "__main__":
    test_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("Linear:")
    for x in test_vals:
        print(f"s={x:.2f} -> alpha={map_style_alpha(x, mode='linear'):.4f}")

    print("\nSigmoid:")
    for x in test_vals:
        print(f"s={x:.2f} -> alpha={map_style_alpha(x, mode='sigmoid', beta=6.0):.4f}")