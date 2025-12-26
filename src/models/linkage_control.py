import torch
import torch.nn.functional as F

class LinkageController:
    """
    Controller for managing TripletAttention layers within a model.
    Provides methods to adjust linkage strength, reset memory, and diagnose mechanism.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.triplet_layers = self._find_triplet_layers()
        print(f"LinkageController initialized. Found {len(self.triplet_layers)} TripletAttention layers.")
        
    def _find_triplet_layers(self):
        """Finds all TripletAttention modules within the model."""
        layers = []
        for name, module in self.model.named_modules():
            # Check by class name to be robust to import variations
            if "TripletAttention" in str(module.__class__):
                layers.append(module)
        return layers

    def triplet_call_counts(self):
        return [
            int(layer._forward_calls.item()) if hasattr(layer, "_forward_calls") else None
            for layer in self.triplet_layers
        ]

    # ... (other methods like set_linkage_strength remain, but we are replacing from set_linkage_strength onwards to cover diagnose and generate)

    def set_linkage_strength(self, value):
        """Sets the linkage strength for all identified TripletAttention layers."""
        for layer in self.triplet_layers:
            layer.set_linkage(value)
            
    def get_linkage_strength(self):
        """Returns the current linkage strength values for all layers."""
        if not self.triplet_layers:
            return []
        return [layer.linkage_strength.item() for layer in self.triplet_layers]
        
    def reset_all_continuity(self):
        """Resets the continuity memory for all identified TripletAttention layers."""
        for layer in self.triplet_layers:
            layer.reset_continuity()

    @torch.no_grad()
    def diagnose_mechanism(self, prompt, topk=10):
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        self.reset_all_continuity()
        self.set_linkage_strength(1.0)
        out1 = self.model(**inputs, use_cache=False)
        logits1 = out1.logits[:, -1, :].float()

        self.reset_all_continuity()
        self.set_linkage_strength(0.0)
        out0 = self.model(**inputs, use_cache=False)
        logits0 = out0.logits[:, -1, :].float()

        diff = logits1 - logits0
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()

        p1 = F.softmax(logits1, dim=-1)
        p0 = F.softmax(logits0, dim=-1)
        kl_10 = (p1 * (p1.clamp_min(1e-12).log() - p0.clamp_min(1e-12).log())).sum().item()

        top1_1 = torch.argmax(logits1, dim=-1).item()
        top1_0 = torch.argmax(logits0, dim=-1).item()

        print("--- PROBE RESULTS ---")
        print(f"Max |Δlogit|: {max_abs:.6f}")
        print(f"Mean |Δlogit|: {mean_abs:.6f}")
        print(f"KL(P1||P0): {kl_10:.6f}")
        print(f"Top-1 same: {top1_1 == top1_0} ({self.tokenizer.decode([top1_1])!r} vs {self.tokenizer.decode([top1_0])!r})")

        return {
            "max_abs_logit_diff": max_abs,
            "mean_abs_logit_diff": mean_abs,
            "kl_next_token(P1||P0)": kl_10,
            "top1_token_same": (top1_1 == top1_0),
        }

    @torch.no_grad()
    def generate_with_linkage(self, prompt, max_tokens=60, linkage_mode="full",
                              do_sample=True, temperature=0.7, top_p=0.9, **kwargs):
        self.model.eval()

        if linkage_mode == "full":
            actual_value = 1.0
        elif linkage_mode == "observer":
            actual_value = 0.0
        elif isinstance(linkage_mode, (float, int)):
            actual_value = float(linkage_mode)
        else:
            raise ValueError(f"Unknown linkage_mode: {linkage_mode}")

        self.reset_all_continuity()
        self.set_linkage_strength(actual_value)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Print start message if not streaming (streaming handles its own output)
        if "streamer" not in kwargs:
            print(f"Generating {linkage_mode} response (max {max_tokens}, temp={temperature})...")

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs 
        )

        full_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text

        return response, {
            "linkage_mode": linkage_mode,
            "actual_linkage": actual_value,
            "num_triplet_layers": len(self.triplet_layers),
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p
        }
