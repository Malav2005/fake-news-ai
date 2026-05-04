import requests

print("Sending request...")

url = "http://127.0.0.1:5000/predict"

data = {
    "text": "Government announces new education policy"
}

response = requests.post(url, json=data)

print("Response received:")
print(response.json())