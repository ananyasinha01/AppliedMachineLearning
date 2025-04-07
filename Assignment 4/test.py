import os
import time
import joblib
import requests
import subprocess
import pytest
from score import score

# Load the model once for all tests
model = joblib.load("model.pkl")
threshold = 0.5

def test_score_endpoint_returns_valid_prediction():
    text = "Congratulations Wow!! 7839 Win a free iPhone now!!!"
    pred, prob = score(text, model, threshold)

    assert isinstance(pred, bool)
    assert 0.0 <= prob <= 1.0


def test_docker_container_response():
    # Build Docker image
    subprocess.run(["docker", "build", "-t", "spam-flask-app", "."], check=True)

    # Run Docker container
    container = subprocess.Popen(
        ["docker", "run", "-p", "5001:5001", "--rm", "spam-flask-app"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        # Wait for Flask app to start
        time.sleep(5)

        # Send POST request to /score endpoint
        response = requests.post(
            "http://localhost:5001/score",
            json={"text": "Congratulations Wow!! 789003 Win a free iPhone now!!!"}
        )

        assert response.status_code == 200
        result = response.json()

        assert "prediction" in result
        assert result["prediction"] in [0, 1]

    finally:
        container.terminate()
        try:
            container.wait(timeout=5)
        except subprocess.TimeoutExpired:
            container.kill()