import requests

if __name__ == '__main__':
    response = requests.post("http://localhost:8000/predict", json={"path": "./images/test_image.png"})
    print(response.json())
