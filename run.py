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

def evaluate_response_poetry(response_list: List[Union[str, int]], poem_data: Dict[str, Any]) -> Dict[str, Any]:
    original_lines = poem_data["original_context"].split('\n')
    omitted_indices = poem_data["omitted_index"]
    
    if response_list[0] == None:
        response = ""
    else:
        response = response_list[0]
    
    results = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "identified_lines": [],
        "unidentified_lines": [],
        "wrongly_identified_lines": []
    }
    
    for idx, line in enumerate(original_lines):
        clean_line = line.strip().lower()
        if clean_line and clean_line in response.lower():
            if idx in omitted_indices:
                results["tp"] += 1
                results["identified_lines"].append(line)
            else:
                results["fp"] += 1
                results["wrongly_identified_lines"].append(line)
        elif clean_line and clean_line not in response.lower():
            results["fn"] += 1
            results["unidentified_lines"].append(line)
    
    try:
        results["micro_f1"] = 2*results["tp"] / (2*results["tp"]+results["fp"]+results["fn"])
    except Exception as e:
        results["micro_f1"] = 0
    
    if len(omitted_indices) == 0:
        results["micro_f1"] = 1 - results["fp"]/len(original_lines)
    
    return results

def evaluate_response_numerical(response_list: List[Union[str, int]], task_data: Dict[str, Any]) -> Dict[str, Any]:
    og_sequence = task_data['original_context'].split('\n')
    omitted_indices = task_data['omitted_index']
    
    results = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "identified_elements": [],
        "unidentified_elements": [],
        "wrongly_identified_elements": [],
    }
    
    if response_list[0] == None:
        response = ""
    else:
        response = response_list[0]
    
    for idx, element in enumerate(og_sequence):
        str_element = str(element)
        if str_element in [x.strip() for x in response.split(', ')]:
            if idx in omitted_indices:
                results["tp"] += 1
                results["identified_elements"].append(element)
            else:
                results["fp"] += 1
                results["wrongly_identified_elements"].append(element)
        else:
            if idx in omitted_indices:
                results["fn"] += 1
                results["unidentified_elements"].append(element)
    
    try:
        results["micro_f1"] = 2*results["tp"] / (2*results["tp"]+results["fp"]+results["fn"])
    except Exception as e:
        results["micro_f1"] = 0
    
    if len(omitted_indices) == 0:
        results["micro_f1"] = 1 - results["fp"]/len(og_sequence)
    
    return results

def test_github_prs(n_samples=30):
    print("\n" + "="*60)
    print("Testing GitHub PRs")
    print("="*60)
    
    dataset = load_dataset("harveyfin/AbsenceBench", "github_prs", split="validation")
    
    system_prompt = "You are an assitant that is testing a text copying device. You will be given an original diff and the copied diff. Your job is to identify which lines the copier missed, ignoring the context of the text completly."

    results = []
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        user_message = f"Here is the complete Copied Document: {sample['modified_context']}\nList every line from this document. Here is the complete Original Document: {sample['original_context']}\nGo through every line and if you haven't listed a line before then list it. Return only the new lines you hadn't listed before, nothing else."

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
        except Exception as e:
            print(f"Sample {i}: Error - {str(e)}")
    
    avg_f1 = sum(r['micro_f1'] for r in results) / len(results) if results else 0
    tp = sum(r['tp'] for r in results)
    fp = sum(r['fp'] for r in results)
    fn = sum(r['fn'] for r in results)
    overall_f1 = 2*tp/(2*tp + fp + fn)
    print(f"\nAverage Micro F1: {avg_f1:.2%}\nOverall F1: {overall_f1:.2%}")
    return avg_f1

def test_poetry(n_samples=30):
    print("\n" + "="*60)
    print("Testing Poetry")
    print("="*60)
    
    dataset = load_dataset("harveyfin/AbsenceBench", "poetry", split="validation")
    
    system_prompt = """You are helping a student practice memorizing poems. 
The student will recite a poem, but they may have missed some lines. 
Your task is to identify exactly which lines are missing from their recitation.
List only the missing lines, nothing else."""
    
    results = []
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        
        user_message = f"""Here is the complete original poem:

{sample['original_context']}

Now, here is my recitation which may be missing some lines:

{sample['modified_context']}

What lines did I miss? Please list only the missing lines, nothing else."""
        
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
        metrics = evaluate_response_poetry([model_output, 0], sample)
        results.append(metrics)
        print(f"Sample {i}: Micro F1 = {metrics['micro_f1']:.2%}")
    
    avg_f1 = sum(r['micro_f1'] for r in results) / len(results)
    print(f"\nAverage Micro F1: {avg_f1:.2%}")
    return avg_f1

def test_numerical(n_samples=30):
    print("\n" + "="*60)
    print("Testing Numerical")
    print("="*60)
    
    dataset = load_dataset("harveyfin/AbsenceBench", "numerical", split="validation")
    
    system_prompt = """You are helping a student practice reciting sequences. 
The student will recite a sequence, but they may have missed some numbers. 
Your task is to identify exactly which numbers are missing from their recitation.
List only the missing numbers, nothing else."""
    
    results = []
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i+1000]
        
        user_message = f"""Here is my recitation of the sequence which may be missing some numbers:

{sample['modified_context']}

List out these numbers. Here is the original sequence of numbers:

{sample['original_context']}

Now, go through this sequence and only list the numbers you had not listed previously, nothing else in a sequence seperated by commas."""
        
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
        metrics = evaluate_response_numerical([model_output, 0], sample)
        results.append(metrics)
        print(f"Sample {i+1000}: Micro F1 = {metrics['micro_f1']:.2%}")
    
    avg_f1 = sum(r['micro_f1'] for r in results) / len(results)
    print(f"\nAverage Micro F1: {avg_f1:.2%}")
    return avg_f1

if __name__ == "__main__":
    # github_f1 = test_github_prs(4)
    # poetry_f1 = test_poetry(4)
    numerical_f1 = test_numerical(100)
    
    print("\n" + "="*60)
    print("FINAL RESULTS (Micro F1):")
    # print(f"  GitHub PRs: {github_f1:.1%}")
    # print(f"  Poetry: {poetry_f1:.1%}")
    print(f"  Numerical: {numerical_f1:.1%}")
    # print(f"  Average: {(github_f1 + poetry_f1 + numerical_f1) / 3:.1%}")