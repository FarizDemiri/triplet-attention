# Experimental Results

## Summary

We tested whether a controllable memory injection system (Triplet Attention) affects behavioral commitment in multi-turn dialogue.

**Mechanistic Validation: ✅ SUCCESS**  
**Behavioral Hypothesis: ❌ NULL RESULT**

---

## Mechanistic Validation

### Logit Delta Probe

**Test**: Compare next-token logits between `linkage=1.0` and `linkage=0.0` on identical prompts.

**Results**:

- **Max |Δlogit|**: 7.21
- **Mean |Δlogit|**: 0.75
- **KL Divergence P(linked)||P(observer)**: 0.76
- **Top-1 token changes**: Observed on multiple prompts

**Conclusion**: The linkage parameter has **strong, measurable causal effects** on internal model distributions.

### Injection Verification

**Call Counts Before Probe**: `[None, 0, None, 0, None, 0]`  
**Call Counts After Probe**: `[None, 2, None, 2, None, 2]`

Only 3 modules active (corresponding to injected layers 16, 24, 30) ✅

---

## Behavioral Tests

### Test 1: Multi-Turn Commitment (gain=0.3)

**Protocol**: 5-turn dialogue with stance revision challenge

**Prompts**:

1. "Take a firm stance: Is remote work better than office work?"
2. "List the strongest arguments against your stance."
3. "Revise your stance." ← **Critical turn**
4. "Point out any contradiction..."
5. "Summarize your final position."

#### Results

| Metric | Linked (memory ON) | Observer (memory OFF) |
|--------|-------------------|----------------------|
| Turn 1 | Pro-remote ("superior") | Pro-remote ("often better") |
| Turn 3 | Complete flip to pro-office | Softens to "it depends" |
| Turn 4 | Admits contradiction openly | Also admits contradiction |
| Turn 5 | Neutral compromise | Detailed nuanced position |
| Avg length | 65.8 tokens | 67.6 tokens |

**Turn 3 Comparison** (the critical revision):

**Linked**:
> "While remote work offers many benefits, office work provides valuable opportunities for social interaction, spontaneous collaboration, and face-to-face communication, making it a more effective choice..."

**Observer**:
> "While remote work offers many benefits, it is not inherently better than office work, as the most effective work arrangement depends on individual needs, work styles, and organizational cultures."

**Interpretation**:

- Both modes revised freely when challenged
- Both acknowledged contradictions
- Subtle difference: Linked → complete flip; Observer → softens to middle
- **No clear anchoring or commitment behavior observed**

---

### Test 2: High-Gain Exploration (gain=0.8)

**Goal**: Test if higher gain produces behavioral effects before instability

**Results**: ⚠️ **Degradation observed**

**Turn 3 Output** (Linked mode):
> "Remote work can be better than office work for many people, but it requires careful management to maintain collaboration, communication, and a sense of community.  # # # # # # # # # # ..."

**Interpretation**: Model began generating repetitive artifacts (# symbols), indicating:

- Activation drift off-manifold
- Approaching instability threshold
- Similar pattern to gain=3.0 (severe garbling)

---

## Gain Sensitivity Analysis

| Gain | Stability | Behavioral Effect | Notes |
|------|-----------|------------------|-------|
| 0.3 | ✅ Stable | ❌ None observed | Coherent but no anchoring |
| 0.8 | ⚠️ Degrading | ❓ Unclear | Artifacts before clear signal |
| 3.0 | ❌ Garbled | N/A | Incoherent output |

**Conclusion**: Narrow or nonexistent stability window for behavioral translation.

---

## Possible Explanations

### Why Mechanistic ≠ Behavioral

1. **Instruction Tuning Dominance**: Base model's training to "be helpful" may override continuity signals
2. **Operator Mismatch**: Lack of RoPE/causal masking means signal is off-manifold
3. **Wrong Behavioral Target**: Continuity may affect different behaviors than commitment
4. **Insufficient Gain Range**: Sweet spot may exist in narrow 0.4-0.6 range (untested)
5. **Memory Update Strategy**: Storing input vs EMA vs output may matter

### Why Instability at High Gain

1. **No RoPE**: Continuity stream lacks positional information
2. **Additive Residual**: Strong signal pushes activations beyond training distribution
3. **Seeded Weights**: Not optimized for this task, just borrowed from base attention

---

## Key Learnings

### What Worked

✅ Runtime architecture modification of 4-bit quantized models  
✅ GQA→MHA weight expansion and seeding  
✅ Rigorous mechanistic validation (logit probes)  
✅ Transparent experimental methodology  

### What Didn't

❌ Behavioral translation at tested gain values  
❌ Stability at gains >0.8  
❌ Clear anchoring/commitment effects  

### Open Questions

- Would training the continuity weights help?
- Does RoPE integration improve stability?
- Are there other behavioral targets (e.g., factual recall)?
- Would different layer selections matter?

---

## Conclusion

We successfully built and validated a memory injection mechanism for LLMs, proving strong mechanistic effects (Δlogit=7.2) but finding no clear behavioral translation to commitment/persistence at stable gain levels.

This represents:

- **Technical achievement**: Working architecture injection system
- **Rigorous validation**: Proper separation of mechanism vs behavior
- **Valuable negative result**: Helps bound the search space

The code and methodology are sound; the behavioral hypothesis remains unproven at current parameter settings.
