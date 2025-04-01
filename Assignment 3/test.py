import pytest
import joblib
import requests
import subprocess
import time
import socket
import sys
from score import score
from app import app as flask_app

# Fixtures for shared test resources
@pytest.fixture(scope="module")
def loaded_model():
    return joblib.load('model.pkl')

@pytest.fixture(scope="module")
def loaded_vectorizer():
    return joblib.load('vectorizer.pkl')

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

# Helper Functions
def is_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

# Unit Tests
def test_score_smoke(loaded_model):
    """Test basic functionality without crashing"""
    prediction, propensity = score("Free money!!!", loaded_model, 0.5)
    assert prediction in [True, False]
    assert 0 <= propensity <= 1

def test_score_format(loaded_model):
    """Verify output types"""
    prediction, propensity = score("Hello world", loaded_model, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)

def test_score_thresholds(loaded_model):
    """Test threshold boundary conditions"""
    # Threshold 0 should always predict True
    assert score("Any text", loaded_model, 0.0)[0] is True
    
    # Threshold 1 should always predict False
    assert score("Any text", loaded_model, 1.0)[0] is False

def test_score_spam_detection(loaded_model):
    """Test obvious spam samples"""
    spam_samples = [
        "WINNER!! Claim your prize now!",
        "1st wk FREE! Gr8 tones str8 2 u each wk. Txt NOKIA ON to 8007 for Classic Nokia tones or HIT ON to 8007 for Polys",
        "congratulations, call 97898090998 claim prize"
    ]
    for text in spam_samples:
        prediction, propensity = score(text, loaded_model, 0.5)
        assert prediction is True, (
            f"Failed on: {text}\n"
            f"Prediction: {prediction}, Propensity: {propensity:.4f}\n"
            "Model failed to detect obvious spam"
        )

def test_score_non_spam(loaded_model):
    """Test obvious non-spam samples"""
    ham_samples = [
        "Hi, how are you doing?",
        "Let's meet for coffee tomorrow",
        "Can we reschedule our meeting?"
    ]
    for text in ham_samples:
        prediction, _ = score(text, loaded_model, 0.5)
        assert prediction is False, f"Failed on: {text}"

# Integration Tests
def test_flask_api(client):
    """Test API endpoint through test client"""
    test_cases = [
        ("congratulations, call 97898090998 claim prize", True),
        ("Hello colleague", False)
    ]
    
    for text, expected in test_cases:
        response = client.post('/score', json={'text': text})
        assert response.status_code == 200
        data = response.get_json()
        assert data['prediction'] is expected
        assert 0 <= data['propensity'] <= 1

def test_live_server():
    """Test running server with enhanced reliability checks"""
    port = 5001
    if not is_port_available(port):
        pytest.skip(f"Port {port} is not available")

    # Start server with explicit python path
    proc = subprocess.Popen(
        [sys.executable, 'app.py'],  # Use same Python interpreter
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={'FLASK_ENV': 'development'}
    )
    
    try:
        # Wait for server with timeout
        for _ in range(10):  # 10 attempts with 0.5s delay
            if not is_port_available(port):
                break
            time.sleep(0.5)
        else:
            pytest.fail("Server failed to start within 5 seconds")
        
        # Test with very obvious spam
        response = requests.post(
            f'http://localhost:{port}/score',
            json={'text': 'WIN $1000 CASH NOW! LIMITED TIME OFFER!', 'threshold': 0.5},
            timeout=5
        )
        assert response.status_code == 200, f"Unexpected status: {response.status_code}"
        data = response.json()
        assert data['prediction'] is True, (
            f"Failed to detect obvious spam\n"
            f"Response: {data}\n"
            f"Threshold: 0.5"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        # Print server output for debugging
        print("\nServer output:")
        print(proc.stdout.read().decode())
        print("Server errors:")
        print(proc.stderr.read().decode())