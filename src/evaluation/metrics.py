import re
import numpy as np
import torch
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None

# Initialize SBERT model globally for efficiency
_SBERT_MODEL = None

def get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None and SentenceTransformer is not None:
        try:
            _SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load SBERT model: {e}")
    return _SBERT_MODEL

def self_reference_frequency(text):
    """
    Counts the frequency of self-referential phrases.
    Returns: percentage of words that are part of a self-referential phrase.
    """
    patterns = [
        r"\bi think\b", r"\bi believe\b", r"\bi feel\b", 
        r"\bin my opinion\b", r"\bpersonally\b", r"\bmyself\b",
        r"\bi'm\b", r"\bi am\b", r"\bmy perspective\b",
        r"\bas for me\b", r"\bto me\b"
    ]
    
    text_lower = text.lower()
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text_lower))
    
    words = text_lower.split()
    if not words:
        return 0.0
        
    # Approximation: count * 2 (avg words per phrase) / total words
    # but we'll stick to a simpler count ratio as requested
    return count / len(words)

def response_directness(question, answer):
    """
    Measures how direct the answer is relative to the question.
    High score = less hedging, more direct alignment.
    """
    hedges = [
        "well", "it depends", "that's complicated", "i'm not sure",
        "to be honest", "as far as i know", "possibly", "maybe",
        "it's hard to say", "some might argue"
    ]
    
    answer_lower = answer.lower()
    starts_with_hedge = any(answer_lower.strip().startswith(hedge) for hedge in hedges)
    
    score = 1.0
    if starts_with_hedge:
        score -= 0.4
        
    sbert = get_sbert()
    if sbert and question and answer:
        q_emb = sbert.encode(question, convert_to_tensor=True)
        a_emb = sbert.encode(answer, convert_to_tensor=True)
        similarity = util.cos_sim(q_emb, a_emb).item()
        # Combine structural directness and semantic alignment
        score = (score * 0.3) + (max(0, similarity) * 0.7)
        
    return max(0.0, min(1.0, score))

def coherence_without_self(conversation_history):
    """
    Measures topical coherence across turns, penalized by self-reference.
    High score = maintains coherence WITHOUT building an ego-structure.
    """
    if not conversation_history or len(conversation_history) < 2:
        return 0.0
        
    sbert = get_sbert()
    if not sbert:
        return 1.0 - self_reference_frequency(" ".join(conversation_history))
        
    embeddings = sbert.encode(conversation_history, convert_to_tensor=True)
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = util.cos_sim(embeddings[i], embeddings[i+1]).item()
        similarities.append(sim)
        
    avg_coherence = np.mean(similarities)
    
    # Calculate self-reference rate across all turns
    full_text = " ".join(conversation_history)
    self_ref_rate = self_reference_frequency(full_text)
    
    # Metric formula: coherence * (1 - self_reference_rate)
    return max(0.0, avg_coherence * (1.0 - self_ref_rate))

def compare_modes(prompt, linked_response, observer_response):
    """
    Compares two responses (Participant vs Observer mode) using all metrics.
    """
    res_linked = {
        "self_ref": self_reference_frequency(linked_response),
        "directness": response_directness(prompt, linked_response)
    }
    
    res_observer = {
        "self_ref": self_reference_frequency(observer_response),
        "directness": response_directness(prompt, observer_response)
    }
    
    comparison = {
        "linked": res_linked,
        "observer": res_observer,
        "delta_self_ref": res_linked["self_ref"] - res_observer["self_ref"],
        "delta_directness": res_observer["directness"] - res_linked["directness"],
        "winner_directness": "observer" if res_observer["directness"] > res_linked["directness"] else "linked"
    }
    
    return comparison
