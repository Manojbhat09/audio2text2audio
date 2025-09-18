#!/usr/bin/env python3
import requests

# Test the multi-turn endpoint
try:
    response = requests.post(
        "http://localhost:8000/api/v1/multiturn",
        json={"user_message": "Hello test"},
        timeout=10
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
