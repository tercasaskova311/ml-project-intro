import json
import requests
import os

# CONFIGURAZIONE
submission_file = "submissions/sub_clip.json"  # percorso relativo al file
groupname = "Stochastic thr"
url = "http://tatooine.disi.unitn.it:3001/retrieval/"  # server di submission

# CARICA IL FILE JSON
if not os.path.exists(submission_file):
    raise FileNotFoundError(f"❌ File non trovato: {submission_file}")

with open(submission_file, "r") as f:
    images = json.load(f)

# PREPARA IL PAYLOAD
payload = {
    "groupname": groupname,
    "images": images
}

# INVIA LA RICHIESTA
try:
    res = json.dumps(payload)
    response = requests.post(url, data=res, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    result = response.json()
    print(f"✅ Submission completed.\nTop-k Accuracy = {result['accuracy']}")
except requests.exceptions.RequestException as e:
    print(f"❌ Network or server error: {e}")
except json.JSONDecodeError:
    print("❌ Response is not valid JSON.")
    print(response.text)