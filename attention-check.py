# proper_attention_analysis.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import numpy as np

# Use Mistral-7B (free on HF, better than GPT-2)
model_name = "mistralai/Mistral-7B-v0.1"
print(f"Loading {model_name}...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    output_attentions=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("harveyfin/AbsenceBench", "poetry", split="validation")
sample = dataset[0]

# Get full context, not truncated
original_lines = sample['original_context'].split('\n')[:30]
omitted_indices = sample['omitted_index']

# Create modified with actual gaps
modified_lines = [line for i, line in enumerate(original_lines) if i not in omitted_indices]

# Create version with explicit markers
marked_lines = []
for i, line in enumerate(original_lines):
    if i in omitted_indices:
        marked_lines.append("<GAP>")
    else:
        marked_lines.append(line)

# Process all three
texts = {
    'original': '\n'.join(original_lines),
    'modified': '\n'.join(modified_lines),
    'marked': '\n'.join(marked_lines)
}

results = {}
for name, text in texts.items():
    inputs = tokenizer(text, return_tensors="pt", max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get attention from middle layers (more interpretable)
    attn = outputs.attentions[len(outputs.attentions)//2][0].mean(0).cpu().numpy()
    
    # Find attention to gap regions
    gap_attention = []
    for idx in omitted_indices[:5]:  # First 5 gaps
        if idx < len(attn):
            gap_attention.append(attn[idx].mean())
    
    results[name] = {
        'avg_gap_attention': np.mean(gap_attention) if gap_attention else 0,
        'total_entropy': -np.sum(attn * np.log(attn + 1e-10))
    }

print("\n=== HARD RESULTS ===")
for name, res in results.items():
    print(f"{name}: Gap attention={res['avg_gap_attention']:.4f}, Entropy={res['total_entropy']:.1f}")

print(f"\nGap attention change:")
print(f"  Modified vs Original: {(results['modified']['avg_gap_attention']/results['original']['avg_gap_attention']-1)*100:.1f}%")
print(f"  Marked vs Modified: {(results['marked']['avg_gap_attention']/results['modified']['avg_gap_attention']-1)*100:.1f}%")