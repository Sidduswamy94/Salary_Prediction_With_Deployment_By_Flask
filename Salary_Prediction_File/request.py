import requests
import numpy as np

url = 'http://localhost:5000/predict_api'
input_data = np.array([4, 9, 7]).reshape(1, -1)  # Reshape the input to (1, 3)
r = requests.post(url, json={'experience': input_data.tolist()})  # Convert to list for JSON serialization

print(r.json())
