import json
import requests
import os

submission_file = "submissions/sub_clip_test.json" 
groupname = "Stochastic thr"
url = "http://tatooine.disi.unitn.it:3001/retrieval/"


if not os.path.exists(submission_file):
    raise FileNotFoundError(f"File not found: {submission_file}")

with open(submission_file, "r") as f:
    images = json.load(f)

payload = {
    "groupname": groupname,
    "images": images
}

try:
    res = json.dumps(payload)
    response = requests.post(url, data=res, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    result = response.json()
    print(f"Submission completed.\nTop-k Accuracy = {result['accuracy']}")
except requests.exceptions.RequestException as e:
    print(f"Network or server error: {e}")
except json.JSONDecodeError:
    print("Response is not valid JSON.")
    print(response.text)