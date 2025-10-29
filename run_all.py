import os
import json
from together import Together # type: ignore
from dotenv import load_dotenv # type: ignore

with open('diff_tests/run_experiment.py', 'r') as file:
    diff_tests = file.read()

with open('paragraph_tests/run_experiment.py', 'r') as file:
    para_tests = file.read()

print("——————————————————————————————————\nDiff tests:\n")
exec(diff_tests)
print("——————————————————————————————————\nParagraph tests:\n")
exec(para_tests)
print("——————————————————————————————————")