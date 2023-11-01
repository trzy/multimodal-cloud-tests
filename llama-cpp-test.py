import base64
import requests

url = "http://127.0.0.1:8080/completion"
params = { 
    "prompt": "Hello, can you hear me?"
}
response = requests.post(url, json = params)
print(response.text)

