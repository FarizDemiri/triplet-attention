# Triplet Attention: Memory Persistence in LLMs

An experimental architecture for injecting controllable memory streams into transformer language models to study the relationship between internal state persistence and behavioral commitment.

## Overview

This project implements a novel attention mechanism that adds a "continuity stream" to Llama 3.1 8B, creating a controllable memory buffer that persists across tokens. We test whether this architectural change affects model behavior in multi-turn dialogue, specifically whether maintaining stronger memory of prior outputs creates behavioral anchoring or commitment.

## Core Innovation

**TripletAttention Module**: A custom attention layer injected into specific Llama layers that:

- Maintains a detached memory buffer of previous hidden states
- Applies cross-attention (Q=current, K/V=memory)
- Gates the signal via a learnable `linkage` parameter (0.0-1.0)
- Returns a residual that's added to the original attention output

**Key Design**: When `linkage=0.0`, the architecture is mathematically identical to the base model (zero residual). When `linkage=1.0`, the continuity signal is fully enabled.

## Technical Achievements ✅

### Architecture

- Successfully injected custom attention into 4-bit quantized Llama 3.1 8B
- Target layers: 16, 24, 30 (late-middle to semantic layers)
- Additive residual design preserves base model when disabled
- Seeded continuity weights from Llama's GQA attention (adapted to MHA format)

### Validation

**Logit Delta Probe** (mechanistic verification):

- Max |Δlogit|: **7.21** (strong effect)
- Mean |Δlogit|: **0.75**
- KL Divergence P(linked)||P(observer): **0.76**
- Result: Linkage parameter **causally affects** next-token distributions

**Call Count Verification**: Confirmed custom layers are on-path during generation

### Engineering

Resolved complex compatibility issues:

- 4-bit weight dequantization (`bitsandbytes` Params4bit handling)
- GQA→MHA weight expansion (8 KV heads → 32 heads)
- Zero-collapse prevention (memory initialization strategy)
- KV cache propagation for generation speed
- Dtype/device alignment across quantized layers

## Behavioral Hypothesis (Under Test) ⚠️

**Question**: Does memory persistence → behavioral persistence?

**Test Protocol**: Multi-turn stance revision

1. Model takes a position
2. User challenges with counter-arguments  
3. Model asked to revise stance ← **critical turn**
4. User asks about contradictions
5. Model summarizes final position

**Prediction**:

- `linkage=1.0`: Should resist full revision, maintain commitment to initial stance
- `linkage=0.0`: Should revise more freely

**Status**: Final tests in progress. Results will be published in `RESULTS.md`.

## Installation

```bash
# Clone repository
git clone https://github.com/farizdemiri/triplet-attention.git
cd triplet-attention

# Install dependencies
pip install -r requirements.txt

# Note: Requires ~11GB GPU VRAM for 4-bit Llama 3.1 8B
```

## Quick Start

```python
from src.models.model_injection import create_triplet_model

# Load model with Triplet layers
model, tokenizer, controller = create_triplet_model()

# Test with linkage enabled
response, _ = controller.generate_with_linkage(
    "Who are you?",
    linkage_mode="full",  # 1.0
    max_tokens=50
)

# Compare with linkage disabled
response_baseline, _ = controller.generate_with_linkage(
    "Who are you?",
    linkage_mode="observer",  # 0.0
    max_tokens=50
)
```

## Run Diagnostic Tests

```bash
# Mechanistic validation (logit probe)
jupyter notebook notebooks/03_implementation_test.ipynb

# Behavioral experiments
jupyter notebook notebooks/06_final_clean_test.ipynb
```

## Project Structure

```
triplet-attention/
├── src/
│   ├── models/
│   │   ├── triplet_attention.py    # Core continuity module
│   │   ├── model_injection.py      # Layer wrapping logic
│   │   └── linkage_control.py      # Experiment controller
│   └── evaluation/
│       └── metrics.py               # Diagnostic tools
├── notebooks/
│   ├── 03_implementation_test.ipynb  # Logit probe results
│   └── 06_final_clean_test.ipynb     # Behavioral experiments
└── README.md
```

## Key Parameters

- **`continuity_gain`**: Scaling factor for continuity signal (currently 0.3)
- **`linkage_strength`**: Gate value 0.0-1.0 controlling memory influence
- **Target layers**: `[16, 24, 30]` for semantic/stylistic effects

## Known Limitations

1. **Operator Mismatch**: Continuity attention doesn't apply RoPE (Rotary Position Embeddings) or causal masking like native Llama attention
2. **High Gain Instability**: Values >1.0 cause off-manifold activations and garbled output
3. **Compute Cost**: ~2-3x slower generation due to extra attention operations
4. **Behavioral Translation**: Strong mechanistic effects don't guarantee semantic/behavioral changes

## Future Directions

- [ ] RoPE integration for continuity stream
- [ ] Attention masking for proper causality
- [ ] Cross-layer memory sharing
- [ ] Training-time optimization (currently inference-only)
- [ ] Smaller models for faster iteration

## Citation

If you use this work, please cite:

```bibtex
@software{triplet_attention_2025,
  title={Triplet Attention: Experimental Memory Persistence in LLMs},
  author={Fariz Demiri},
  year={2025},
  url={https://github.com/farizdemiri/triplet-attention}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built using:

- Hugging Face Transformers
- Meta's Llama 3.1
- bitsandbytes (4-bit quantization)
- PyTorch

---

**Status**:

- Mechanistic validation: ✅ Complete (logit Δ=7.21, KL=0.76)
- Behavioral testing: ✅ Complete (see [RESULTS.md](RESULTS.md))
- Conclusion: Strong mechanistic effects; behavioral translation unclear

---

**Transparency Note**: This is experimental research. The mechanistic validation is solid (logit probe confirms causal effects), but behavioral effects are still under investigation. We're publishing this work-in-progress to contribute to the open-source exploration of memory mechanisms in LLMs.
