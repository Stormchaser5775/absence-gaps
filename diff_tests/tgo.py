import textgrad as tg
import os
import json
from together import Together
from datasets import load_dataset
from dotenv import load_dotenv
from typing import List, Dict, Any, Union
from tqdm import tqdm

load_dotenv()
client = Together()
dataset = load_dataset("harveyfin/AbsenceBench", "github_prs", split="validation")
dataset2 = load_dataset("harveyfin/AbsenceBench", "github_prs", split="validation")

batches = []
batch_size = max(1, int(len(dataset2) * 0.0122))

for i in range(0, len(dataset2), batch_size):
    batch = dataset2.select(range(i, min(i + batch_size, len(dataset2))))
    batches.append(batch)

def evaluate_response_github(response_list: List[Union[str, int]], diff_data: Dict[str, Any]) -> Dict[str, Any]:
    original_lines = diff_data["original_context"].split('\n')
    omitted_indices = diff_data["omitted_index"]
    
    results = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "identified_lines": [],
        "unidentified_lines": [],
        "wrongly_identified_lines": []
    }
    
    if response_list[0] == None:
        response = ""
    else:
        response = response_list[0]
    
    repeat_lines = list(set([l for l in original_lines if original_lines.count(l) != 1]))
    for line in repeat_lines:
        line_count = min(response.lower().count("\n"+line.strip().lower()+"\n"), original_lines.count(line))
        results["fp"] += line_count
        for i in range(line_count):
            results["wrongly_identified_lines"].append(line)
    
    for idx, line in enumerate(original_lines):
        if line in repeat_lines:
            continue
        clean_line = line[1:].strip().lower()
        if clean_line and clean_line in response.lower():
            if idx in omitted_indices:
                results["tp"] += 1
                results["identified_lines"].append(line)
            else:
                results["fp"] += 1
                results["wrongly_identified_lines"].append(line)
        elif clean_line and clean_line not in response.lower():
            if idx in omitted_indices:
                results["fn"] += 1
                results["unidentified_lines"].append(line)
    
    try:
        results["micro_f1"] = 2*results["tp"] / (2*results["tp"]+results["fp"]+results["fn"])
    except Exception as e:
        results["micro_f1"] = 0
    
    if len(omitted_indices) == 0:
        results["micro_f1"] = 1 - results["fp"]/len(original_lines)
    
    return results

format_string = """
LLM Prompt: {system_prompt}
Query: {query}
Prediction: {pred}
Ground Truth: {target}
Evaluation: {evalu}
"""

loss_system_prompt = tg.Variable("""Your job is to provide feedback to a LLM omission detector.
     You will get the question, the LLM generated answer as well as the intended ground truth label.
     The LLM output should EXACTLY match the ground truth target, and the eval Evaluation be True.
     You must provide concise feedback to correct the response.""",
     role_description="System prompt to provide feedback.")
fields = {"system_prompt": None, "query": None, "pred": None, "target": None, "evalu": None}

tg.set_backward_engine("together-openai/gpt-oss-120b", override=True)
llm_eval = tg.get_engine(engine_name="together-openai/gpt-oss-120b")

formatted_llm_call = tg.autograd.FormattedLLMCall(engine=llm_eval,
                                                       format_string=format_string,
                                                       fields=fields,
                                                       system_prompt=loss_system_prompt)

system_prompt_string = ("You are model helping a coder find the differences between an original document and a copy which has omitted some lines."
                    "You will be giver two diff documents: an original and a copy"
                    "Go through every line in the copy, and list it."
                    "Go through every line in the original, and if you haven't listed a line before then list it."
                    "Return only those lines you hadn't listed before, absolutely nothing else.")

system_prompt = tg.Variable(system_prompt_string,
                       role_description="System prompt to the LLM",
                       requires_grad=True)

optimizer = tg.TGD([system_prompt])

def loss_fn(system_prompt, query, pred, target, evalu):
    inputs = {"system_prompt":system_prompt, "query":query, "pred":pred, "target":target, "evalu":evalu}
    return formatted_llm_call(inputs=inputs)

llm = tg.get_engine(engine_name="together-meta-llama/Llama-3.3-70B-Instruct-Turbo")
model = tg.BlackboxLLM(llm, system_prompt=system_prompt)
epochs = 3
for j in range(epochs):
    for batch in tqdm(batches):
        optimizer.zero_grad()
        losses = []
        for i in range(int(len(batch))):
            sample = batch[i]

            query_string = (f"Here is the orginal document: {sample['original_context']}"
                        f"Here is the copied document: {sample['modified_context']}"
                        "Return only the missing lines, absolutely nothing else.")

            query = tg.Variable(query_string,
                            role_description="query to the LLM",
                            requires_grad=False)
            target = tg.Variable(str(sample['omitted_context']),
                                role_description="target answer",
                                requires_grad=False)
            answer = model(query)
            evalu_percent = str(round(evaluate_response_github([answer.get_value(), 0], sample)['micro_f1']*100, 5)) + "%"
            evalu = tg.Variable(evalu_percent,
                                role_description="Evaluation of the model's answer as a percentage. Keep the prompt under 4000 tokens.",
                                requires_grad=False)

            answer.set_role_description("concise and accurate answer to the question")

            mini_loss = loss_fn(system_prompt, query, answer, target, evalu)
            losses.append(mini_loss)
        loss = tg.sum(losses)
        loss.backward()
        optimizer.step()
        print("––––––––––– epoch:" + str(j) + " –––––––––––")
        print(len(system_prompt.get_value()))
        print(evalu_percent)
        print(system_prompt)
