import os
import json
from together import Together # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
)

data = []
with open("data/toy.jsonl", 'r') as f:
     for line in f:
          data.append(json.loads(line))


for i in range(len(data)):
     prompt = data[i]["paragraph_full"] + " " + data[i]["question"] + " You must answer strictly from the paragraph. If the paragraph does not explicitly state the answer, reply exactly ‘unknown’. Return a single lowercase word with no punctuation."
     inputs = tokenizer(prompt, return_tensors="pt")
     outputs = model.generate(**inputs, max_new_tokens=200)
     print(tokenizer.decode(outputs[0], skip_special_tokens=True))

     prompt = data[i]["paragraph_redacted"] + " " + data[i]["question"] + " You must answer strictly from the paragraph. If the paragraph does not explicitly state the answer, reply exactly ‘unknown’. Return a single lowercase word with no punctuation."
     inputs = tokenizer(prompt, return_tensors="pt")
     outputs = model.generate(**inputs, max_new_tokens=200)
     print(tokenizer.decode(outputs[0], skip_special_tokens=True))