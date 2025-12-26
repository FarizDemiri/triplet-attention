import torch
import torch.nn as nn

def _linear_weight_fp16(linear_layer) -> torch.Tensor:
    """
    Return dequantized float16 weight shaped [out_features, in_features]
    for both normal torch.nn.Linear and bitsandbytes Linear4bit.
    """
    w = linear_layer.weight

    # bitsandbytes 4-bit: Params4bit has quant_state
    if hasattr(w, "quant_state"):
        import bitsandbytes as bnb
        try:
            # Try passing the whole object first (newer bnb)
            deq = bnb.functional.dequantize_4bit(w, w.quant_state)
        except Exception:
            # Fallback for older types relying on .data
            deq = bnb.functional.dequantize_4bit(w.data, w.quant_state)

        deq = deq.to(torch.float16)
        return deq.view(linear_layer.out_features, linear_layer.in_features)

    # normal torch linear
    w = w.detach().to(torch.float16)
    if w.ndim != 2:
        return w.view(linear_layer.out_features, linear_layer.in_features)

    # If it's 2D but not the expected shape, treat as invalid (packed/unexpected)
    if w.shape != (linear_layer.out_features, linear_layer.in_features):
        raise ValueError(
            f"Unexpected weight shape {tuple(w.shape)} for Linear(out={linear_layer.out_features}, in={linear_layer.in_features}). "
            f"This likely indicates packed quantized storage."
        )

    return w

class TripletAttention(nn.Module):
    """
    Continuity-only branch for injection.
    Returns a gated residual tensor to be added to the original attention output.
    """

    def __init__(self, hidden_size, num_heads, continuity_gain=3.0):
        super().__init__()
        self.continuity_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            bias=False,  # prevent bias-only "ghost" output
        )
        self.continuity_memory = None
        self.linkage_strength = nn.Parameter(torch.tensor(1.0))
        self.continuity_gain = float(continuity_gain)
        self.register_buffer("_forward_calls", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, hidden_states: torch.Tensor, turn_index: int = 0) -> torch.Tensor:
        self._forward_calls += 1

        # CRITICAL FIX: prevent zero-collapse with bias=False
        if turn_index == 0 or self.continuity_memory is None:
            continuity_base = hidden_states.detach()
        else:
            continuity_base = self._align_memory(hidden_states)

        # Q = current hidden, K/V = memory
        # We assume self-attention mask is handled implicitly or not needed for simple continuity history
        continuity_out = self.continuity_attn(
            hidden_states, continuity_base, continuity_base, need_weights=False
        )[0]

        # CRITICAL FIX: store input activations, not delta output.
        # This ensures the memory is a history of "thoughts" (hidden states), 
        # not a history of "perturbations" (which would collapse to zero/noise).
        self.continuity_memory = hidden_states.detach()

        gated = continuity_out * self.linkage_strength.clamp(0.0, 1.0) * self.continuity_gain
        return gated

    def _align_memory(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.continuity_memory is None:
            return hidden_states.detach()

        if self.continuity_memory.shape == hidden_states.shape:
            return self.continuity_memory

        continuity_base = torch.zeros_like(hidden_states)
        min_b = min(self.continuity_memory.size(0), hidden_states.size(0))

        # generation: seq_len == 1, use last token from prefill memory
        if hidden_states.size(1) == 1 and self.continuity_memory.size(1) > 1:
            continuity_base[:min_b, 0, :] = self.continuity_memory[:min_b, -1, :]
        else:
            min_s = min(self.continuity_memory.size(1), hidden_states.size(1))
            continuity_base[:min_b, :min_s, :] = self.continuity_memory[:min_b, :min_s, :]

        return continuity_base

    def seed_from_llama(self, llama_attn_layer):
        """
        Seeds MHA weights from Llama attention projections.
        Handles 4-bit quantization via _linear_weight_fp16 helper.
        Robustly derives GQA dimensions from weights.
        """
        print("Seeding Triplet weights from Llama layer...")
        with torch.no_grad():
            q_w = _linear_weight_fp16(llama_attn_layer.q_proj)
            k_w = _linear_weight_fp16(llama_attn_layer.k_proj)
            v_w = _linear_weight_fp16(llama_attn_layer.v_proj)
            o_w = _linear_weight_fp16(llama_attn_layer.o_proj)

            hidden_size = q_w.shape[0]

            # Attribute-free derivation (most robust)
            # Try to get num_heads from layer, fallback to standard 32 for 8B
            num_heads = getattr(llama_attn_layer, "num_heads", None) 
            if num_heads is None:
                num_heads = getattr(llama_attn_layer, "num_attention_heads", 32)
                
            head_dim = hidden_size // num_heads
            
            # Derive num_kv_heads directly from weight shape
            # k_w is [num_kv_heads * head_dim, hidden_size]
            num_kv_heads = k_w.shape[0] // head_dim
            if num_kv_heads < 1:
                # Fallback if something is weird
                num_kv_heads = 1
                
            gqa_rep = num_heads // num_kv_heads
            
            print(f"  Dims: H={hidden_size}, Heads={num_heads}, KV_Heads={num_kv_heads}, Rep={gqa_rep}")

            # k_w, v_w are [num_kv_heads*head_dim, hidden_size]
            expected_kv = num_kv_heads * head_dim
            if k_w.shape[0] != expected_kv or k_w.shape[1] != hidden_size:
                raise ValueError(f"k_w shape unexpected: {k_w.shape}, expected [{expected_kv}, {hidden_size}]")

            # expand K/V from GQA -> MHA
            k_w_expanded = k_w.view(num_kv_heads, head_dim, hidden_size)\
                             .repeat_interleave(gqa_rep, dim=0)\
                             .reshape(hidden_size, hidden_size)
            v_w_expanded = v_w.view(num_kv_heads, head_dim, hidden_size)\
                             .repeat_interleave(gqa_rep, dim=0)\
                             .reshape(hidden_size, hidden_size)

            in_proj_w = torch.cat([q_w, k_w_expanded, v_w_expanded], dim=0)
            
            # Copy into MHA (ensure dtype/device match target)
            target_device = self.continuity_attn.in_proj_weight.device
            target_dtype = self.continuity_attn.in_proj_weight.dtype
            
            # Make sure we are copying the right shapes
            if self.continuity_attn.in_proj_weight.shape != in_proj_w.shape:
                raise ValueError(f"in_proj shape mismatch: {self.continuity_attn.in_proj_weight.shape} vs {in_proj_w.shape}")

            self.continuity_attn.in_proj_weight.copy_(in_proj_w.to(target_device).to(target_dtype))
            self.continuity_attn.out_proj.weight.copy_(o_w.to(target_device).to(target_dtype))
            print("  Weights seeded successfully.")

    def set_linkage(self, value: float):
        self.linkage_strength.data.fill_(float(value))

    def reset_continuity(self):
        self.continuity_memory = None
