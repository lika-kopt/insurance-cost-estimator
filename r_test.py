
import requests


url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}
ls = [55.0, 29.83, 0.0]
data = {"input": ls}
r = requests.get(url, headers=headers, json=data)
print(r.text)

