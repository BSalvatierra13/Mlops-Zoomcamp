import requests

ride = {
    "PUlocationID": 151,
    "DOlocationID": 151,
    "trip_distance": 3.75
}

url = 'http://localhost:9696/predict'

response = requests.post(url, json=ride)
print(response.json())