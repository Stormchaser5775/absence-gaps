import os
import json
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
               answer = eval(answer)
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
               accuracy = str(numCorrect/len(targets[index]) * 100)
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

outputFile = open("diff_tests/outputs/diff_outputs.jsonl", "w+") # Empties the file
outputFile.close()
for i in range(len(data)):
     oData = data[i]
     cData = modData[i]

     systemPrompt = "You are helping a software developer determine if their merge of a pull request was successful. The developer had to edit the commit history and just wants to make sure that they have not changed what will be merged. They will list the changed lines. Your job is to figure out if they have missed any insertions or deletions from the original merge. Only pay attention to the insertions and deletions (ignore the context of the diff)."

     userPrompt1 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else. There are always less than 10 answers."

     userPrompt2 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else. There are always less than 10 answers. Search the whole document."
     
     # userPrompt3 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? To answer this, compare line by line and find the lines without matching pairs. List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else."

     # userPrompt4 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? To answer this, divide each document into lists one new lines, compare line by line, and find the lines without matching pairs. Double check that the missing lines actually exist in the original document. There are only 5 missing lines. List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else. Search the whole document."

     answerLists = askModel(systemPrompt, [userPrompt1, userPrompt2], models)

     outputer(answerLists, i)