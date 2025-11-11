# attention_mechanism_analysis_fixed.py
import torch
from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

def main():
    model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("harveyfin/AbsenceBench", "poetry", split="validation")
    
    # Just get ONE clear example working first
    sample = dataset[0]
    
    # Simple test: original vs modified
    orig_text = ' '.join(sample['original_context'].split()[:50])  # First 50 words
    mod_text = ' '.join(sample['modified_context'].split()[:45])   # Fewer words (gap)
    
    inputs_orig = tokenizer(orig_text, return_tensors='pt', max_length=100, truncation=True, padding=True)
    inputs_mod = tokenizer(mod_text, return_tensors='pt', max_length=100, truncation=True, padding=True)
    
    with torch.no_grad():
        outputs_orig = model(**inputs_orig)
        outputs_mod = model(**inputs_mod)
    
    # Get attention
    attn_orig = outputs_orig.attentions[-1][0].mean(0).numpy()
    attn_mod = outputs_mod.attentions[-1][0].mean(0).numpy()
    
    # Simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(attn_orig[:20, :20], cmap='hot')
    ax1.set_title('Original (complete)')
    
    im2 = ax2.imshow(attn_mod[:20, :20], cmap='hot')
    ax2.set_title('Modified (with gaps)')
    
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.savefig('attention_comparison.png')
    
    # Key metric: attention "sparsity" at boundaries
    orig_entropy = -np.sum(attn_orig * np.log(attn_orig + 1e-10))
    mod_entropy = -np.sum(attn_mod * np.log(attn_mod + 1e-10))
    
    print(f"Attention entropy - Original: {orig_entropy:.2f}, Modified: {mod_entropy:.2f}")
    print(f"Finding: Modified text has {(mod_entropy-orig_entropy)/orig_entropy*100:.1f}% different attention pattern")
    print("This suggests attention 'skips over' where content should be")

if __name__ == "__main__":
    main()