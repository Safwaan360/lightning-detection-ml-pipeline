"""Small API smoke test. Start app.py before running this file."""
import requests

payload = {
    "id": 13,
    "starttime": "2025-10-31 18:46:50",
    "data": [1024] * 728,
    "model": "hybrid",
}

response = requests.post("http://localhost:5000/predict", json=payload, timeout=10)
response.raise_for_status()
print(response.json())
