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
    device_map="auto"
)

data = []
with open("data/toy.jsonl", 'r') as f:
     for line in f:
          data.append(json.loads(line))


for i in range(len(data)):
     data[i]["question"]