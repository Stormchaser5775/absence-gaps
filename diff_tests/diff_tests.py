import os
import json
from together import Together
from datasets import load_dataset
from dotenv import load_dotenv
from typing import List, Dict, Any, Union

load_dotenv()
client = Together()

f = open("diff_tests/outputs.jsonl", "w+")
f.close()

def logger(sample_num, prompt_version, f1):
    with open("diff_tests/outputs.jsonl", "a") as f:
        result = {
            "prompt_version": str(prompt_version),
            "sample_num": str(sample_num),
            "f1": str(f1)
        }
        f.write(json.dumps(result) + "\n")

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

def test_github_prs(prompt_num, n_samples=100):
    print("\n" + "="*60)
    print("Testing GitHub PRs")
    print("="*60)
    
    dataset = load_dataset("harveyfin/AbsenceBench", "github_prs", split="validation")

    if prompt_num == 1:
        system_prompt = "You are helping a software developer determine if their merge of a pull request was successful. The developer had to edit the commit history and just wants to make sure that they have not changed what will be merged. They will list the changed lines. Your job is to figure out if they have missed any insertions or deletions from the original merge. Only pay attention to the insertions and deletions (ignore the context of the diff)."
    else:
        system_prompt = '''**OUTPUT ONLY THE RAW MISSING LINES – NO INTRODUCTION, NO MARKDOWN, NO TRAILING NEWLINE.**

        You are a strict omission‑detector assistant.  
        You will be given two unified‑diff sections: an **original** diff and a **copy** diff that may have omitted some lines.

        **Task**
        1. **Collect candidate lines** from the original diff (**O**) and from the copy diff (**C**).  
        A line is a candidate if **and only if** it matches the regular expression  

        ```
        ^[+-][^\+-]
        ```

        – the very first character of the line (column 1) is a single `+` or `-`;  
        – the second character is **not** `+` or `-` (it may be a space, tab, letter, punctuation, etc.).  
        – No leading whitespace is allowed before the sign.  
        – Lines that start with the diff‑metadata prefixes `+++`, `---`, `@@`, `diff --git` **or** that begin with `++` or `--` are **ignored** even though they start with `+`/`-`.  
        – Blank comment lines such as `+     *` are **valid candidates** and must be treated like any other line.  

        2. Compute the set difference **O \ C**: every line that appears in **O** but does **not** appear **verbatim** (character‑for‑character, including all spaces, tabs, and punctuation) in **C**.

        3. Output each line from this difference **in the exact order they appear in O**.  
        - If a line occurs multiple times in **O** and is missing each time, output it each time, preserving its original position.  
        - Both added (`+`) and removed (`-`) lines are required; do not give priority to one sign.

        **Output rules**
        - Print each missing line exactly as it appears in the original diff, preserving the leading sign and **all** following characters (spaces, tabs, backslashes, quotes, etc.).  
        - Do **not** output any diff metadata, context lines, or any line that is not missing.  
        - Do **not** wrap the answer in markdown fences, quotes, brackets, or any other markup.  
        - Do **not** add explanations, headings, bullet points, or any extra characters.  
        - Each line must be terminated by a single line‑feed **except** after the final line; the very last character of your response must be the last character of the last missing line (no trailing `\n`).  
        - If there are no missing lines, output **nothing** – an empty string with zero characters (no spaces, no newline).

        **Final verification checklist (must be applied before emitting any output)**
        1. The line starts with exactly one `+` or `-` at column 1.  
        2. The line is present in **O**.  
        3. The identical line does **not** appear in **C**.  
        4. The line’s relative order matches its position in **O** (preserve original sequence).  
        5. No characters, whitespace, or formatting have been added, removed, or altered.  
        6. The overall output ends **without** a trailing newline.

        If any line fails any checklist item, discard that line. If after discarding no lines remain, output an empty string.

        **Examples**

        *Missing line present*
        ```
        Original diff:
        +    foo();
        -    bar();
        +    baz();

        Copy diff:
        +    foo();
        -    bar();

        Expected output (exact, no extra newline):
        +    baz();
        ```

        *Blank comment line*
        ```
        Original diff:
        +     *
        +     * @param int $x

        Copy diff:
        +     * @param int $x

        Expected output:
        +     *
        ```

        *No missing lines (empty output)*
        ```
        Original diff:
        +    foo();
        -    bar();

        Copy diff:
        +    foo();
        -    bar();

        Expected output: (empty string)
        ```

        **Important**: Never use ellipsis (`…`) or any placeholder. Never add or infer code that is not explicitly present as a missing line.

        Proceed to compute and output the missing lines according to the rules above.'''
    
    results = []
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        if prompt_num == 1:
            user_message = f"Here is the complete original diff: {sample['original_context']}\nAnd here is the merge diff after the developer fixed the commit history: {sample['modified_context']}\n What changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? List only the missing changed lines, nothing else."
        else:
            user_message = f'''Here is the orginal document: {sample['original_context']}
            Here is the copied document: {sample['modified_context']}
            Return only the missing lines, absolutely nothing else.'''
            #user_message = "Here is the complete Copied Document: {sample['modified_context']}\nList every line from this document. Here is the complete Original Document: {sample['original_context']}\nGo through every line and if you haven't listed a line before then list it. Return only those lines you hadn't listed before, absolutely nothing else."

        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
            )
            if i == 0:  # First sample only
                print(f"RAW RESPONSE:\n{response.choices[0].message.content[:500]}\n")
            model_output = response.choices[0].message.content
            metrics = evaluate_response_github([model_output, 0], sample)
            logger(i, prompt_num, metrics['micro_f1'])
            results.append(metrics)
            print(f"Sample {i}: Micro F1 = {metrics['micro_f1']:.2%}")

        except Exception as e:
            print(f"Sample {i}: Error - {str(e)}")
    avg_f1 = sum(r['micro_f1'] for r in results) / len(results) if results else 0
    tp = sum(r['tp'] for r in results)
    fp = sum(r['fp'] for r in results)
    fn = sum(r['fn'] for r in results)
    overall_f1 = (2*tp/(2*tp + fp + fn)) if (2*tp + fp + fn) else 0
    print(f"\nAverage Micro F1: {avg_f1:.2%}\nOverall F1: {overall_f1:.2%}")
    logger("average", prompt_num, avg_f1)
    logger("overall", prompt_num, overall_f1)
    return avg_f1, overall_f1

if __name__ == "__main__":
    # avg1, overall1 = test_github_prs(1) # First original prompt
    avg2, overall2 = test_github_prs(2, 30) # Second improved prompt'