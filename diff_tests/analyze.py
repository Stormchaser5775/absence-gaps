import json

data = []

with open("diff_tests/outputs1.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

nums1 = data[:100]
nums2 = data[102:202]
results = []
for i in range(len(nums1)):
     results.append(abs(float(nums2[i]['f1'])-float(nums1[i]['f1'])))

avg = 0
for i in results:
    avg += i
avg /= len(results)
print(avg)