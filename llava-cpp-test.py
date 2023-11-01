import base64
import requests

with open("car.jpg", "rb") as fp:
    data = fp.read()
image_data = base64.b64encode(data).decode("utf-8")


url = "http://127.0.0.1:8080/completion"
params = { 
    "prompt": "[img-0]USER: What color is the vehicle? Describe in ONE WORD only.\nASSISTANT:",
    "image_data": [
        { "data": image_data, "id": 0 }
    ]
}
response = requests.post(url, json = params)
print(response.text)

