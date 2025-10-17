import os
import json
from together import Together # type: ignore

data = []
with open("data/toy.jsonl", 'r') as f:
     for line in f:
          data.append(json.loads(line))

# for i in data:
#      data[i][""]