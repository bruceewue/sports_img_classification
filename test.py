import requests

# url  = "http://localhost:8080/2015-03-31/functions/function/invocations"
url = "https://qmvv6ahdp9.execute-api.eu-west-2.amazonaws.com/test/predict/"
data = {"url": "https://i.ibb.co/tYRBfFR/2.jpg"}

result = requests.post(url, json=data).json()
print(result)
