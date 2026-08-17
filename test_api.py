import urllib.request
import json

# Test tanpa weekNumber (seperti yang dikirim frontend)
data = json.dumps({
    "ipk": 3.0,
    "attendanceRate": 40,
    "assignmentScore": 50,
    "quizScore": 50,
    "atsScore": 30
}).encode()
req = urllib.request.Request("http://127.0.0.1:8001/predict", data=data, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    print("Tanpa weekNumber (atsScore=30):", json.loads(resp.read()))

# Test tanpa weekNumber dan tanpa atsScore
data2 = json.dumps({
    "ipk": 3.0,
    "attendanceRate": 40,
    "assignmentScore": 50,
    "quizScore": 50
}).encode()
req2 = urllib.request.Request("http://127.0.0.1:8001/predict", data=data2, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req2) as resp:
    print("Tanpa weekNumber, tanpa atsScore:", json.loads(resp.read()))

# Test dengan weekNumber
data3 = json.dumps({
    "weekNumber": 4,
    "ipk": 3.0,
    "attendanceRate": 40,
    "assignmentScore": 50,
    "quizScore": 50,
    "atsScore": 30
}).encode()
req3 = urllib.request.Request("http://127.0.0.1:8001/predict", data=data3, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req3) as resp:
    print("Dengan weekNumber=4:", json.loads(resp.read()))
