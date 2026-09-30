"""Optional local PostgreSQL integration test; credentials come from environment variables."""
import os
import psycopg2
import requests

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_NAME", "lightning"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()
cur.execute(
    "SELECT id, detector, data, rtsecs FROM sample "
    "WHERE detector IN (13, 17, 18) ORDER BY RANDOM() LIMIT 1"
)
row = cur.fetchone()
if row is None:
    raise RuntimeError("No rows found in local sample table")

sample_id, detector, raw_data, rtsecs = row
waveform = raw_data if isinstance(raw_data, list) else list(map(int, raw_data.strip("{}").split(",")))

response = requests.post(
    "http://localhost:5000/predict",
    json={"id": sample_id, "starttime": str(rtsecs), "data": waveform},
    timeout=10,
)
response.raise_for_status()
print({"detector": detector, **response.json()})
cur.close()
conn.close()
