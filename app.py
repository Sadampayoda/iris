from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model once at the start to avoid loading it in each request
with open('models/iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/load_model')
def load_model():
    try:
        with open('models/iris_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return "Model loaded successfully!"
    except Exception as e:
        return str(e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        probabilities = model.predict_proba(features).tolist()  # Get prediction probabilities
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
