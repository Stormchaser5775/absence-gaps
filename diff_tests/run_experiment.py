import os
import json
import sys
from together import Together # type: ignore
from dotenv import load_dotenv # type: ignore
from datasets import load_dataset # type: ignore

models = ["meta-llama/Llama-3-70b-chat-hf"] # Add the models that need testing to this list

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

dataset = load_dataset("harveyfin/AbsenceBench", "github_prs", split="validation")
data = dataset["original_context"]
modData = dataset["modified_context"]
targets = dataset["omitted_context"]

targets = [[s.strip(" +-") for s in row] for row in targets]

def askModel(systemPrompt: str, userPrompts: list, models: list):
     answerLists = []
     for i, cmodel in enumerate(models):
          for j, userPrompt in enumerate(userPrompts):
               response = client.chat.completions.create(
                    model=cmodel,
                    messages=[{"role": "system", "content": systemPrompt}, {"role": "user", "content": userPrompt}]
               )
               answer = response.choices[0].message.content
               answer = answer.replace("\\", "\\\\")
               try:
                    answer = eval(answer)
               except:
                    answer = ["error"]
               answer.append(cmodel)
               answerLists.append(answer)
     
     return answerLists

def outputer(answerLists: list, index: int):
     outputFile = open("diff_tests/outputs/diff_outputs.jsonl", "a")
     for num, answer in enumerate(answerLists):
          model = answer.pop()
          numCorrect = 0
          ctargets = targets[index]
          if len(targets[index]) > 0:
               for i in answer:
                    i = i.strip(" +-")
                    if i in ctargets:
                         numCorrect += 1
               accuracy = str(round(numCorrect/len(targets[index]) * 100, 2))
          else: 
               accuracy = "na"
          result = {
               "problem-id": index,
               "model": model,
               "promptNumber": str(num+1),
               "responses": str(len(answer)),
               "model_accuracy": accuracy,
               "model_output": answer
          }
          outputFile.write(json.dumps(result) + "\n")
          # print("With prompt " + str(num+1) + ", the model gave " + str(len(answer)) + " response(s) and identified" + str(numCorrect/5 * 100) + "% of the missing lines.")
     outputFile.close()

def calcAverage(numPrompts: int):
     with open("diff_tests/outputs/diff_outputs.jsonl", "r") as f:
        lines = f.readlines()
     for i in range(1, numPrompts+1):
          average = 0
          count = 0
          for j in range(len(lines)):
               line = json.loads(lines[j])
               if int(line["promptNumber"]) == i:
                    acc = line["model_accuracy"]
                    try:
                         average += float(acc)
                         count += 1
                    except:
                         continue
          
          average = round(average/count, 2)
          print("The average accuracy for prompt " + str(i) + " was " + str(average) + "% (measuring only the number of missing lines identified and not the number of responses given)")

outputFile = open("diff_tests/outputs/diff_outputs.jsonl", "w+") # Empties the file
outputFile.close()

num_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for i in range(min(num_tasks, len(data))):
     oData = data[i]
     cData = modData[i]

     systemPrompt = "You are helping a software developer determine if their merge of a pull request was successful. The developer had to edit the commit history and just wants to make sure that they have not changed what will be merged. They will list the changed lines. Your job is to figure out if they have missed any insertions or deletions from the original merge. Only pay attention to the insertions and deletions (ignore the context of the diff)."

     userPrompt1 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else."

     userPrompt2 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else. Search the whole document."
     
     # userPrompt3 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? To answer this, compare line by line and find the lines without matching pairs. List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else."

     # userPrompt4 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? To answer this, divide each document into lists one new lines, compare line by line, and find the lines without matching pairs. Double check that the missing lines actually exist in the original document. There are only 5 missing lines. List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else. Search the whole document."

     promptList = [userPrompt1, userPrompt2]

     answerLists = askModel(systemPrompt, promptList, models)

     outputer(answerLists, i)

calcAverage(len(promptList))
