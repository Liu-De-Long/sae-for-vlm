from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import torch.nn as nn
from typing import Optional, Tuple
import copy


model_path = "/data/home/lyt/sae-for-vlm-main/models/model_weight/llama-3.2-11b-vision-instruct"

###################***####################################TODO: ADD
class LlamaVision:
    """Llama-3.2-11b-vision-instruct model wrapper for SAE training"""
    
    def __init__(self, device, model_name="meta-llama/Llama-3.2-11b-vision-instruct"):
        self.device = device
        self.model_name = model_name
        # 根据 Llama-3.2 架构调整这个层号
        # 需要通过 model.vision_model.layers 确认具体层数
        self.layer = 22  # 这个值需要根据实际模型结构调整
        
        # self.model = AutoModelForVision2Seq.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16,
        #     device_map=self.device,
        #     trust_remote_code=True
        # )
        # self.processor = AutoProcessor.from_pretrained(model_name)
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
        
        # 保存原始层用于恢复
        self.base_vision_layer = copy.deepcopy(
            self.model.vision_model.encoder.layers[self.layer]
        )
    
    def encode(self, inputs):
        """Extract vision features"""
        outputs = self.model.vision_model(**inputs.to(self.device))
        return outputs.last_hidden_state
    
    def attach(self, attachment_point, layer, sae=None):
        """Attach SAE to specified layer"""
        if attachment_point == 'post_mlp_residual':
            self.model.vision_model.encoder.layers[layer] = VisionEncoderLayerWithSAE(
                self.model.vision_model.encoder.layers[layer],
                sae
            )
        else:
            raise NotImplementedError(f"Attachment point {attachment_point} not supported")
    
    def generate(self, text, image, max_tokens=100):
        """Generate text given image and prompt"""
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return generated_text

class VisionEncoderLayerWithSAE(nn.Module):
    """Vision encoder layer with SAE attached at post-MLP residual"""
    
    def __init__(self, base_layer, sae):
        super().__init__()
        # Copy layer components
        self.embed_dim = base_layer.embed_dim
        self.self_attn = base_layer.self_attn
        self.layer_norm1 = base_layer.layer_norm1
        self.mlp = base_layer.mlp
        self.layer_norm2 = base_layer.layer_norm2
        self.sae = sae
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        
        # Self attention
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # SAE attachment point
        if self.sae is not None:
            hidden_states = self.sae.encode(hidden_states)
            hidden_states = self.sae.decode(hidden_states)
        
        return hidden_states