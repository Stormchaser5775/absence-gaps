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
     prompt1 = data[i]["paragraph_full"] + " " + data[i]["question2"] + " You must answer strictly from the paragraph. If the paragraph does not explicitly state the answer, reply exactly ‘unknown’. Return a single lowercase word with no punctuation."
     prompt2 = data[i]["paragraph_redacted"] + " " + data[i]["question2"] + " You must answer strictly from the paragraph. If the paragraph does not explicitly state the answer, reply exactly ‘unknown’. Return a single lowercase word with no punctuation."
     
     prompt3 = "Original: " + data[i]["paragraph_full"] + "Copy: " + data[i]["paragraph_redacted"] + " " + data[i]["question2"] + " You must answer strictly from the paragraph. If the paragraphs do not explicitly state the answer, reply exactly ‘unknown’. Return only the exact sentence that is missing and nothing else."
     
     
     # response1 = client.chat.completions.create(
     #    model="meta-llama/Llama-3-70b-chat-hf",
     #    messages=[{"role": "user", "content": prompt1}]
     # )

     # response2 = client.chat.completions.create(
     #      model="meta-llama/Llama-3-70b-chat-hf",
     #      messages=[{"role": "user", "content": prompt2}]
     # )
     
     response3 = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf",
        messages=[{"role": "user", "content": prompt3}]
     )

     # answer1 = response1.choices[0].message.content
     # answer2 = response2.choices[0].message.content
     answer3 = response3.choices[0].message.content

     # print("Full paragraph answer:", answer1)
     # print("Redacted paragraph answer:", answer2)
     print("Answer " + str(i+1) + ": " + answer3)