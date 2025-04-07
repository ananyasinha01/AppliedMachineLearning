from flask import Flask, request, jsonify, render_template_string
import joblib
from score import score
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load models
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle both form and JSON submissions
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        prediction, propensity = score(text, model, 0.5)
        
        if request.is_json:
            return jsonify({
                "prediction": prediction,
                "propensity": propensity
            })
        else:
            return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Spam Classifier</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
                        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
                        .spam { background-color: #ffdddd; color: #d8000c; }
                        .ham { background-color: #ddffdd; color: #4F8A10; }
                    </style>
                </head>
                <body>
                    <h1>Spam Classifier</h1>
                    <form method="post">
                        <textarea name="text" rows="4" style="width:100%"></textarea><br>
                        <button type="submit">Check</button>
                    </form>
                    {% if prediction is not none %}
                    <div class="result {{ 'spam' if prediction else 'ham' }}">
                        <h3>Result: {{ 'SPAM' if prediction else 'Not Spam' }}</h3>
                        <p>Propensity: {{ "%.2f"|format(propensity) }}</p>
                    </div>
                    {% endif %}
                </body>
                </html>
            ''', prediction=prediction, propensity=propensity)
    
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spam Classifier</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
                textarea { width: 100%; height: 100px; }
            </style>
        </head>
        <body>
            <h1>Spam Classifier</h1>
            <form method="post">
                <textarea name="text" placeholder="Enter text to classify..."></textarea><br>
                <button type="submit">Check</button>
            </form>
        </body>
        </html>
    ''')

@app.route('/score', methods=['POST'])
def score_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data['text']
    threshold = data.get('threshold', 0.5)
    
    prediction, propensity = score(text, model, threshold)
    return jsonify({
        "prediction": prediction,
        "propensity": propensity
    })

# Add these test cases to test.py
def test_app_error_handling(client):
    """Test missing text field"""
    response = client.post('/score', json={})
    assert response.status_code == 400
    assert 'error' in response.json()

def test_app_get_request(client):
    """Test GET request to root"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Spam Classifier" in response.data

def test_app_html_submission(client):
    """Test form submission"""
    response = client.post('/', data={'text': 'Test'}, 
                         headers={'Content-Type': 'application/x-www-form-urlencoded'})
    assert response.status_code == 200
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)