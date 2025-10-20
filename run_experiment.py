import os
import json
from together import Together # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

data = []
with open("data/toy.jsonl", 'r') as f:
     for line in f:
          data.append(json.loads(line))


for i in range(len(data)):
     prompt1 = data[i]["paragraph_full"] + " " + data[i]["question"] + " You must answer strictly from the paragraph. If the paragraph does not explicitly state the answer, reply exactly ‘unknown’. Return a single lowercase word with no punctuation."
     prompt2 = data[i]["paragraph_redacted"] + " " + data[i]["question"] + " You must answer strictly from the paragraph. If the paragraph does not explicitly state the answer, reply exactly ‘unknown’. Return a single lowercase word with no punctuation."
     
     response1 = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf",
        messages=[{"role": "user", "content": prompt1}]
     )

     response2 = client.chat.completions.create(
          model="meta-llama/Llama-3-70b-chat-hf",
          messages=[{"role": "user", "content": prompt2}]
     )
     print(response1)