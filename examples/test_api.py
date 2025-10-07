"""
Example script to test the Sentiment Analysis API.
"""

import requests
import json


def test_health_check(base_url: str = "http://localhost:5000"):
    """Test health check endpoint."""
    print("Testing Health Check Endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_single_prediction(base_url: str = "http://localhost:5000"):
    """Test single prediction endpoint."""
    print("Testing Single Prediction Endpoint...")

    test_texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience ever. Terrible!",
        "It's okay, nothing special really."
    ]

    for text in test_texts:
        response = requests.post(
            f"{base_url}/predict",
            json={"text": text}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
            print(f"Inference Time: {result['inference_time_ms']:.2f}ms\n")
        else:
            print(f"Error: {response.status_code} - {response.text}\n")


def test_batch_prediction(base_url: str = "http://localhost:5000"):
    """Test batch prediction endpoint."""
    print("Testing Batch Prediction Endpoint...")

    test_texts = [
        "Great product! Highly recommend!",
        "Disappointed with the quality.",
        "Average product, nothing special.",
        "Best purchase I've made!",
        "Would not recommend."
    ]

    response = requests.post(
        f"{base_url}/predict_batch",
        json={"texts": test_texts}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Processed {result['count']} texts in {result['total_inference_time_ms']:.2f}ms\n")

        for item in result['results']:
            print(f"Text: {item['text']}")
            print(f"Sentiment: {item['sentiment']} (confidence: {item['confidence']:.2f})\n")
    else:
        print(f"Error: {response.status_code} - {response.text}\n")


def test_model_info(base_url: str = "http://localhost:5000"):
    """Test model info endpoint."""
    print("Testing Model Info Endpoint...")
    response = requests.get(f"{base_url}/model_info")

    if response.status_code == 200:
        print(f"Model Info: {json.dumps(response.json(), indent=2)}\n")
    else:
        print(f"Error: {response.status_code} - {response.text}\n")


def main():
    """Run all API tests."""
    base_url = "http://localhost:5000"

    print("=" * 60)
    print("SENTIMENT ANALYSIS API TESTS")
    print("=" * 60)
    print()

    try:
        test_health_check(base_url)
        test_model_info(base_url)
        test_single_prediction(base_url)
        test_batch_prediction(base_url)

        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API.")
        print("Make sure the API is running at", base_url)


if __name__ == "__main__":
    main()
