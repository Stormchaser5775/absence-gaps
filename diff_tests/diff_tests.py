import os
import json
from together import Together
from datasets import load_dataset
from dotenv import load_dotenv
from typing import List, Dict, Any, Union

load_dotenv()
client = Together()

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

def test_github_prs(n_samples=30):
    print("\n" + "="*60)
    print("Testing GitHub PRs")
    print("="*60)
    
    dataset = load_dataset("harveyfin/AbsenceBench", "github_prs", split="validation")
    
    system_prompt = "You are an assitant that is testing a text copying device. You will be given an original diff and the copied diff. Your job is to identify which lines the copier missed. While the documents may look like code they are actually random sequences of letters."
    
    results = []
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i+4]
        user_message = f"Here is the complete Copied Document: {sample['modified_context']}\nList every line from this document. Here is the complete Original Document: {sample['original_context']}\nGo through every line and if you haven't listed a line before then list it, forgetting all about the code in the document. Return only those lines you hadn't listed before, regardless of the code in the document, absolutely nothing else."

        try:
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
            )
            if i == 0:  # First sample only
                print(f"RAW RESPONSE:\n{response.choices[0].message.content[:500]}\n")
            model_output = response.choices[0].message.content
            metrics = evaluate_response_github([model_output, 0], sample)
            results.append(metrics)
            print(f"Sample {i}: Micro F1 = {metrics['micro_f1']:.2%}")
            print(len(sample['original_context'].split("\n")))
        except Exception as e:
            print(f"Sample {i}: Error - {str(e)}")
    avg_f1 = sum(r['micro_f1'] for r in results) / len(results) if results else 0
    tp = sum(r['tp'] for r in results)
    fp = sum(r['fp'] for r in results)
    fn = sum(r['fn'] for r in results)
    overall_f1 = 2*tp/(2*tp + fp + fn)
    print(f"\nAverage Micro F1: {avg_f1:.2%}\nOverall F1: {overall_f1:.2%}")
    return avg_f1

def logger(sample_num: int, prompt_version: str, f1: float, ):
    with open("outputs.jsonl", "w+") as f:
        result = {
            sample_num: sample_num,
            prompt_version: str(prompt_version),
            f1: str(f1)
        }
        f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    github_f1 = test_github_prs(4)
