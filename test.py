import requests
import random
import json
import time

# Define the API endpoint
url = "http://localhost:5000/api/update_data"

# Define the vehicle types
vehicle_types = ["car", "truck", "motorbike", "bus"]

# Function to generate random vehicle data
def generate_random_vehicle_data():
    vehicle_data = []
    for _ in range(random.randint(1, 5)):  # Random number of vehicles (1 to 5)
        vehicle = {
            "type": random.choice(vehicle_types),
            "speed": random.randint(50, 120),  # Random speed between 50 and 120
            "timestamp": f"12:{random.randint(0, 59):02}:{random.randint(0, 59):02}",
            "position": [random.uniform(-180, 180), random.uniform(-90, 90)]  # Random position
        }
        vehicle_data.append(vehicle)
    return vehicle_data

# Continuously send random data to Flask
while True:
    # Generate random data
    data = {"vehicles": generate_random_vehicle_data()}
    
    # Send the data to the Flask server
    response = requests.post(url, json=data)
    
    # Print the response from the server
    print("Response Status:", response.status_code)
    print("Response JSON:", response.json())
    
    # Wait for a random time between 1 and 5 seconds before sending the next request
    time.sleep(random.randint(1, 5))  # Random delay between 1 and 5 seconds
