import requests

# Set your local server URL
url = "http://127.0.0.1:8000"


response_get = requests.get(url + "/")
print("GET Status Code:", response_get.status_code)
print("GET Response:", response_get.json())


sample_data = {
    "age": 45,
    "workclass": "Private",
    "fnlgt": 284582,
    "education": "Doctorate",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States"
}


response_post = requests.post(url + "/data/", json=sample_data)
print("POST Status Code:", response_post.status_code)
print("POST Response:", response_post.json())
