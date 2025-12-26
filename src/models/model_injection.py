import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.models.triplet_attention import TripletAttention
from src.models.linkage_control import LinkageController

class LlamaTripletAttentionWrapper(nn.Module):
    """
    Wrapper for TripletAttention to match the LlamaAttention forward signature.
    Implements ADDITIVE RESIDUAL INJECTION:
    Output = Original_Attn(x) + Triplet_Attn(x)
    """
    def __init__(self, triplet_module, original_self_attn=None):
        super().__init__()
        self.triplet_module = triplet_module
        self.original_self_attn = original_self_attn
        
    def set_linkage(self, value):
        """Delegates linkage setting to the inner triplet module."""
        self.triplet_module.set_linkage(value)
        
    def reset_continuity(self):
        """Delegates memory reset to the inner triplet module."""
        self.triplet_module.reset_continuity()
        
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Any:
        # Determine turn_index for Triplet memory management
        # position_ids is usually in kwargs or early in args
        position_ids = kwargs.get("position_ids")
        if position_ids is None and len(args) > 1:
            # Fallback for positional position_ids (Transformers version dependent)
            position_ids = args[1]
            
        turn_index = 0
        if position_ids is not None and position_ids.numel() > 0:
            turn_index = int(position_ids.min().item())

        # 1. Execute original attention logic (The "Body")
        # Captures the robust, pre-trained content signal.
        if self.original_self_attn is not None:
            attn_out = self.original_self_attn(hidden_states=hidden_states, *args, **kwargs)
            
            if isinstance(attn_out, tuple):
                content_out = attn_out[0]
            else:
                content_out = attn_out
        else:
            # Fallback (unlikely)
            content_out = hidden_states 
            attn_out = (content_out, None)

        # 2. Execute parallel Triplet Continuity stream (The "Ego")
        # Returns the GATED residual tensor directly.
        triplet_delta = self.triplet_module(hidden_states, turn_index=turn_index)
        
        # 3. Additive Residual Injection
        # We ADD the ego signal to the content signal.
        # If linkage=0, triplet_delta is 0, preserving perfect original model fidelity.
        output = content_out + triplet_delta
        
        # 4. Reconstruct original return format (tuple or tensor)
        if isinstance(attn_out, tuple):
            return (output,) + attn_out[1:]
        else:
            return output

def load_base_model(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """Loads Llama-3.1-8B with 4-bit quantization."""
    print(f"Loading base model: {model_name}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def inject_triplet_layers(model, target_layers=[16, 24, 30]):
    """Injects TripletAttention by wrapping the original attention layers."""
    print(f"Injecting TripletAttention into layers: {target_layers}...")
    
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    
    for layer_idx in target_layers:
        original_self_attn = model.model.layers[layer_idx].self_attn
        
        # Initialize triplet module and match the specific layer's device and model's dtype
        # We use q_proj's device to handle multi-GPU sharding correctly.
        layer_device = original_self_attn.q_proj.weight.device
        
        # NOTE: gain=0.8 for final high-gain test
        # Fix: explicitly use float16 to ensure safe weight copying
        triplet_module = TripletAttention(hidden_size, num_heads, continuity_gain=0.8).to(layer_device).to(torch.float16)
        
        # Priority 3: Seed weights from the original model to ensure meaningful signal
        triplet_module.seed_from_llama(original_self_attn)
        
        # Create wrapper that PRESERVES the original attention module
        wrapped_layer = LlamaTripletAttentionWrapper(triplet_module, original_self_attn)
        
        # Replace
        model.model.layers[layer_idx].self_attn = wrapped_layer
        print(f" - Layer {layer_idx}: Successfully wrapped original attention.")
        
    return model

def create_triplet_model(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    model, tokenizer = load_base_model(model_name)
    model = inject_triplet_layers(model)
    controller = LinkageController(model, tokenizer)
    return model, tokenizer, controller
