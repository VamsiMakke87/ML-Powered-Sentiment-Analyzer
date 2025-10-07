"""
Flask REST API for sentiment analysis inference.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from dotenv import load_dotenv

from src.preprocessing import TextPreprocessor
from src.model import SentimentLSTM

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'models/sentiment_model.keras')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', 'models/tokenizer.pkl')

# Global variables
model = None
preprocessor = None


def load_model_and_preprocessor():
    """Load model and preprocessor at startup."""
    global model, preprocessor

    print("Loading model and preprocessor...")
    start_time = time.time()

    try:
        model = SentimentLSTM.load_model(MODEL_PATH, TOKENIZER_PATH)
        preprocessor = TextPreprocessor()

        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Load model at startup
with app.app_context():
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        load_model_and_preprocessor()
    else:
        print("Warning: Model files not found. Please train the model first.")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for a single text.

    Request body:
    {
        "text": "Your text here"
    }

    Returns:
    {
        "text": "Your text here",
        "sentiment": "positive/negative/neutral",
        "confidence": 0.95,
        "probabilities": {
            "negative": 0.02,
            "neutral": 0.03,
            "positive": 0.95
        },
        "inference_time_ms": 12.5
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get request data
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400

        text = data['text']

        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Text must be a non-empty string'}), 400

        # Start timer
        start_time = time.time()

        # Preprocess
        processed_text = preprocessor.preprocess(text)

        # Predict
        predictions = model.predict([processed_text])[0]
        sentiment_class = model.predict_class([processed_text])[0]

        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Prepare response
        response = {
            'text': text,
            'sentiment': sentiment_class,
            'confidence': float(predictions.max()),
            'probabilities': {
                'negative': float(predictions[0]),
                'neutral': float(predictions[1]),
                'positive': float(predictions[2])
            },
            'inference_time_ms': round(inference_time, 2)
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple texts.

    Request body:
    {
        "texts": ["Text 1", "Text 2", ...]
    }

    Returns:
    {
        "results": [
            {
                "text": "Text 1",
                "sentiment": "positive",
                "confidence": 0.95,
                ...
            },
            ...
        ],
        "total_inference_time_ms": 45.6
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get request data
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field in request'}), 400

        texts = data['texts']

        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'Texts must be a non-empty list'}), 400

        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts per request'}), 400

        # Start timer
        start_time = time.time()

        # Preprocess
        processed_texts = preprocessor.preprocess_batch(texts)

        # Predict
        predictions = model.predict(processed_texts)
        sentiment_classes = model.predict_class(processed_texts)

        # Calculate inference time
        total_inference_time = (time.time() - start_time) * 1000

        # Prepare response
        results = []
        for i, (text, sentiment, probs) in enumerate(zip(texts, sentiment_classes, predictions)):
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': float(probs.max()),
                'probabilities': {
                    'negative': float(probs[0]),
                    'neutral': float(probs[1]),
                    'positive': float(probs[2])
                }
            })

        response = {
            'results': results,
            'count': len(results),
            'total_inference_time_ms': round(total_inference_time, 2)
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    info = {
        'vocab_size': model.vocab_size,
        'max_length': model.max_length,
        'embedding_dim': model.embedding_dim,
        'lstm_units': model.lstm_units,
        'model_path': MODEL_PATH
    }

    return jsonify(info), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
