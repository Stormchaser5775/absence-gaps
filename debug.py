import os
import json
from together import Together
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

dataset = load_dataset("harveyfin/AbsenceBench", "github_prs", split="validation")
sample = dataset[0]

prompt = f"""Compare these two diffs and find what's missing.

Original diff:
{sample['original_context'][:500]}...  # Truncated for debugging

Modified diff:
{sample['modified_context'][:500]}...  # Truncated for debugging

List ONLY the missing lines in JSON format: {{"missing": ["line1", "line2", ...]}}"""

print("Ground truth missing lines:")
print(sample['omitted_context'])
print("\nSending to model...")

response = client.chat.completions.create(
    model="meta-llama/Llama-3-70b-chat-hf",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    seed=42
)

print("\nRaw model response:")
print(response.choices[0].message.content)

try:
    parsed = json.loads(response.choices[0].message.content)
    print("\nParsed response:")
    print(parsed)
except Exception as e:
    print(f"\nFailed to parse: {e}")