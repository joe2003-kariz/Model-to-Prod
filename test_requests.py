import requests

API_URL = "http://127.0.0.1:5000/predict"
IMAGE_PATH = r"C:\Users\Student\OneDrive\Desktop\Data Science\From Model to Production\archive\images\1596.jpg"

with open(IMAGE_PATH, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(API_URL, files=files)

print("Status:", response.status_code)
print("Response:", response.json())
