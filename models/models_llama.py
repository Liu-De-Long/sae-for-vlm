try:
    from transformers import AutoProcessor, AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import torch.nn as nn
from typing import Optional, Tuple
import copy

model_path = "/root/lyt_code/sae-for-vlm-main/models/LLaMA-3.2-11B-Vision-Instruct"


class LlamaVision:
    """Llama-3.2-11b-vision-instruct vision encoder wrapper for SAE training"""

    def __init__(self,device,model_name="meta-llama/Llama-3.2-11b-vision-instruct",
        token_mode: str = "last",  # "last" | "all"
    ):
        self.device = device
        self.model_name = model_name
        self.token_mode = token_mode

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=False,
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.register = {}
        self.attach_methods = {
            "post_mlp_residual": self._attach_post_mlp_residual,
        }

        layers = self._get_vision_layers("transformer")

        self.layer = len(layers) - 2

        self.base_vision_layer = copy.deepcopy(layers[self.layer])

    def _get_vision_layers(self, which: str = "transformer"):
        if which == "transformer":
            return self.model.vision_model.transformer.layers
        elif which == "global_transformer":
            return self.model.vision_model.global_transformer.layers
        else:
            raise ValueError("which must be 'transformer' or 'global_transformer'")

    def encode(self, inputs):
        # 清空 register 中的激活
        for hook in list(self.register.keys()):
            self.register[hook] = []
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        outputs = self.model.vision_model(**inputs)
        # outputs = self.model.vision_model(**inputs.to(self.device))
        return outputs.last_hidden_state

    def attach(self, attachment_point, layer=None, sae=None):
        """Attach SAE to specified layer (default: self.layer)."""
        if layer is None:
            layer = self.layer
        if attachment_point in self.attach_methods:
            self.attach_methods[attachment_point](layer, sae)
            self.register[f"{attachment_point}_{layer}"] = []
        else:
            raise NotImplementedError(f"Attachment point {attachment_point} not supported")

    # def _attach_post_mlp_residual(self, layer, sae):
    #     layers = self._get_vision_layers()
    #     layers[layer] = VisionEncoderLayerWithSAE(
    #         base_layer=layers[layer],
    #         sae=sae,
    #         layer=layer,
    #         register=self.register,
    #         token_mode=self.token_mode,
    #     )
    def _attach_post_mlp_residual(self, layer, sae):
        """Attach at post-MLP residual.

        Layer indexing rule (no new CLI args):
          - 0..31  -> vision_model.transformer.layers
          - 32..39 -> vision_model.global_transformer.layers (index = layer-32)
        """
        if layer is None:
            layer = self.layer

        local_layers = self._get_vision_layers("transformer")
        global_layers = self._get_vision_layers("global_transformer")

        if layer < len(local_layers):
            layers = local_layers
            inner_idx = layer
        else:
            inner_idx = layer - len(local_layers)
            if inner_idx < 0 or inner_idx >= len(global_layers):
                raise IndexError(
                    f"Vision layer index out of range: layer={layer}. "
                    f"Valid: 0..{len(local_layers) - 1} (transformer) or "
                    f"{len(local_layers)}..{len(local_layers) + len(global_layers) - 1} (global_transformer)."
                )
            layers = global_layers

        layers[inner_idx] = VisionEncoderLayerWithSAE(
            base_layer=layers[inner_idx],
            sae=sae,
            layer=layer,  # keep *global* layer index for register key
            register=self.register,
            token_mode=self.token_mode,
        )

    def generate(self, text, image, max_tokens=100):
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return generated_text


# class VisionEncoderLayerWithSAE(nn.Module):
#     """
#     Vision encoder layer with SAE attached at post-MLP residual.
#     Supports token_mode:
#       - "all": SAE acts on all image tokens (original behavior)
#       - "last": SAE acts ONLY on last token (global image-level)
#     """
#
#     def __init__(self, base_layer, sae, layer, register, token_mode: str = "last"):
#         super().__init__()
#         # self.embed_dim = base_layer.embed_dim
#         self.self_attn = base_layer.self_attn
#         self.layer_norm1 = base_layer.layer_norm1
#         self.mlp = base_layer.mlp
#         self.layer_norm2 = base_layer.layer_norm2
#
#         self.sae = sae
#         self.layer = layer
#         self.register = register
#         self.token_mode = token_mode
#
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.FloatTensor]:
#         residual = hidden_states
#
#         # Self-attention block
#         hidden_states = self.layer_norm1(hidden_states)
#         hidden_states, attn_weights = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#         )
#         hidden_states = residual + hidden_states
#
#         # MLP block
#         residual = hidden_states
#         hidden_states = self.layer_norm2(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states  # ✅ residual stream (post-MLP residual)
#
#         key = f"post_mlp_residual_{self.layer}"
#
#         if self.token_mode == "last":
#             last_tok = hidden_states[:, -1:, :]  # [B, 1, D]
#
#             # register 存 residual stream 上的 last token（用于训练/缓存）
#             self.register[key].append(last_tok.detach().cpu())
#
#             if self.sae is not None:
#                 z = self.sae.encode(last_tok)      # [B, 1, W] (or similar)
#                 recon = self.sae.decode(z)         # [B, 1, D]
#                 # 把重构写回去，其余 tokens 不动
#                 hidden_states = hidden_states.clone()
#                 hidden_states[:, -1:, :] = recon.to(hidden_states.dtype)
#
#         else:
#             # 原始：对所有 tokens 做 SAE
#             self.register[key].append(hidden_states.detach().cpu())
#             if self.sae is not None:
#                 z = self.sae.encode(hidden_states)
#                 recon = self.sae.decode(z)
#                 hidden_states = recon.to(hidden_states.dtype)
#
#         outputs = (hidden_states,)
#         if output_attentions:
#             outputs += (attn_weights,)
#         return outputs
class VisionEncoderLayerWithSAE(nn.Module):
    """
    Mllama vision encoder layer wrapper with SAE attached at post-MLP residual.
    token_mode:
      - "all": SAE acts on all tokens
      - "last": SAE acts ONLY on last token
    """

    def __init__(self, base_layer, sae, layer, register, token_mode: str = "last"):
        super().__init__()

        # ---- MllamaVisionEncoderLayer compatibility ----
        self.self_attn = getattr(base_layer, "self_attn")
        self.mlp = getattr(base_layer, "mlp")

        # LayerNorm names in Mllama
        self.layer_norm1 = (
            getattr(base_layer, "input_layernorm", None)
            or getattr(base_layer, "layer_norm1", None)
            or getattr(base_layer, "ln_1", None)
        )
        self.layer_norm2 = (
            getattr(base_layer, "post_attention_layernorm", None)
            or getattr(base_layer, "layer_norm2", None)
            or getattr(base_layer, "ln_2", None)
        )
        if self.layer_norm1 is None or self.layer_norm2 is None:
            raise AttributeError(
                f"Unsupported vision layer type: {type(base_layer).__name__}. "
                "Expected attributes like input_layernorm/post_attention_layernorm."
            )

        # infer embed dim (optional)
        try:
            ns = self.layer_norm1.normalized_shape
            self.embed_dim = int(ns[0]) if isinstance(ns, (tuple, list)) else int(ns)
        except Exception:
            try:
                self.embed_dim = int(getattr(self.self_attn.q_proj, "in_features"))
            except Exception:
                self.embed_dim = None

        self.sae = sae
        self.layer = layer  # global index (0..39)
        self.register = register
        self.token_mode = token_mode

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        """Forward compatible with MllamaVisionEncoderLayer.
        We reproduce the layer computation and inject SAE at the post-MLP residual stream.
        Extra kwargs are accepted for forward-compatibility with transformers versions.
        """
        # -------- Self-attention block --------
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        attn_weights = None
        try:
            attn_out = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                **{k: v for k, v in kwargs.items() if k in ["position_ids", "past_key_value", "use_cache"]},
            )
        except TypeError:
            # some versions use positional args
            attn_out = self.self_attn(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)

        if isinstance(attn_out, (tuple, list)):
            hidden_states = attn_out[0]
            if output_attentions and len(attn_out) > 1:
                attn_weights = attn_out[1]
        else:
            hidden_states = attn_out

        hidden_states = residual + hidden_states

        # -------- MLP block --------
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # ✅ post-MLP residual stream

        # -------- SAE tap (post_mlp_residual) --------
        key = f"post_mlp_residual_{self.layer}"

        if self.token_mode == "last":
            last_tok = hidden_states[:, -1:, :]  # [B, 1, D]
            self.register[key].append(last_tok.detach().cpu())

            if self.sae is not None:
                z = self.sae.encode(last_tok)
                recon = self.sae.decode(z)
                hidden_states = hidden_states.clone()
                hidden_states[:, -1:, :] = recon.to(dtype=hidden_states.dtype, device=hidden_states.device)
        else:
            self.register[key].append(hidden_states.detach().cpu())
            if self.sae is not None:
                z = self.sae.encode(hidden_states)
                recon = self.sae.decode(z)
                hidden_states = recon.to(dtype=hidden_states.dtype, device=hidden_states.device)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
