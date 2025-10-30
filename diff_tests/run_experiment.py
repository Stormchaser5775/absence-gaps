import os
import json
from together import Together # type: ignore
from dotenv import load_dotenv # type: ignore

models = ["meta-llama/Llama-3-70b-chat-hf"] # Add the models that need testing to this list

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

data = []
with open("diff_tests/diffData/tests.jsonl", 'r') as f:
     for line in f:
          data.append(json.loads(line))

def askModel(systemPrompt: str, userPrompts: list, models: list):
     answerLists = []
     for i, cmodel in enumerate(models):
          for j, userPrompt in enumerate(userPrompts):
               response = client.chat.completions.create(
                    model=cmodel,
                    messages=[{"role": "system", "content": systemPrompt}, {"role": "user", "content": userPrompt}]
               )
               answer = response.choices[0].message.content
               answer = eval(answer)
               answer.append(cmodel)
               answerLists.append(answer)
     
     return answerLists

def outputer(answerLists, line, LINES):
     outputFile = open("diff_tests/outputs/diff_outputs.jsonl", "a")
     for num, answer in enumerate(answerLists):
          model = answer.pop()
          numCorrect = 0
          targetsCopy = eval(line["targets"])
          for i in answer:
               i = i.strip("-+ ")
               if i in targetsCopy:
                    numCorrect += 1
                    targetsCopy.remove(i)
          outputString = "For the test with " + LINES + " lines and prompt " + str(num+1) + ", " + str(len(answer)) + " response(s) were given and " + str(numCorrect/5 * 100) + "% of the 5 missing lines were identified."
          result = {
               "model": model,
               "promptNumber": str(num+1),
               "Number of lines": LINES,
               "model_accuracy": outputString,
               "model_output": answer
          }
          outputFile.write(json.dumps(result) + "\n")
          print(outputString)
     outputFile.close()

outputFile = open("diff_tests/outputs/diff_outputs.jsonl", "w+") # Empties the file
outputFile.close()
for n, line in enumerate(data):
     LINES = line["id"]
     oDoc = "diff_tests/diffData/" + "test" + LINES + ".diff"
     cDoc = "diff_tests/diffData/" + "test" + LINES + "c" + ".diff"
     
     oData = ""
     cData = ""

     with open(oDoc, 'r') as f:
          oData = f.read()
     
     with open(cDoc, 'r') as f:
          cData = f.read()

     systemPrompt = "You are helping a software developer determine if their merge of a pull request was successful. The developer had to edit the commit history and just wants to make sure that they have not changed what will be merged. They will list the changed lines. Your job is to figure out if they have missed any insertions or deletions from the original merge. Only pay attention to the insertions and deletions (ignore the context of the diff)."

     userPrompt1 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else."

     userPrompt2 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else. Search the whole document."
     
     userPrompt3 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? To answer this, compare line by line and find the lines without matching pairs. List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else."

     # userPrompt4 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? To answer this, divide each document into lists one new lines, compare line by line, and find the lines without matching pairs. Double check that the missing lines actually exist in the original document. There are only 5 missing lines. List only the missing changed lines as a python list with no slash-n new lines and each line in double quotes; nothing else. Search the whole document."

     answerLists = askModel(systemPrompt, [userPrompt1, userPrompt2, userPrompt3], models)

     outputer(answerLists, line, LINES)