import os
import json
from together import Together # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

data = []

with open("diff_tests/diffData/tests.jsonl", 'r') as f:
     for line in f:
          data.append(json.loads(line))

outputFile = open("diff_tests/outputs/diff_outputs.jsonl", "w+")

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

     userPrompt1 = "Here is the complete original diff: " + oData + "\nAnd here is the merge diff after the developer fixed the commit history: " + cData + "\nWhat changed lines (insertions or deletions) present in the original diff are missing in the merge diff (if any)? List only the missing changed lines as python list with no slash-n new lines and each line in double quotes; nothing else."

     response1 = client.chat.completions.create(
          model="meta-llama/Llama-3-70b-chat-hf",
          messages=[{"role": "system", "content": systemPrompt}, {"role": "user", "content": userPrompt1}]
     )
     answer1 = response1.choices[0].message.content
     # print("Answer: " + answer1)
     answerLists = [] # I've used this along with suffixing '1' to a bunch of variäbles in case we want to test different prompts side by side
     answerLists.append(eval(answer1))
     numCorrect = 0
     for i in answerLists:
          for j in i:
               j = j.strip("-+ ")
               if j in line["targets"]:
                    numCorrect += 1
          outputString = "For the test with " + LINES + " lines, " + str(len(i)) + " response(s) were given and " + str(numCorrect/5 * 100) + "% of the 5 missing lines were identified."
          result = {
               "Number of lines": LINES,
               "model_accuracy": outputString
          }
          outputFile.write(json.dumps(result) + "\n")
          print(outputString)
outputFile.close()