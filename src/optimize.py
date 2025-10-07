"""
Model optimization utilities for improved inference performance.
Includes quantization and conversion to TensorFlow Lite.
"""

import tensorflow as tf
import numpy as np
import os
from typing import List
import time


class ModelOptimizer:
    """Optimize TensorFlow models for production inference."""

    def __init__(self, model_path: str):
        """
        Initialize optimizer.

        Args:
            model_path: Path to Keras model
        """
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    def quantize_model(
        self,
        output_path: str,
        quantization_type: str = 'dynamic'
    ) -> str:
        """
        Quantize model to reduce size and improve inference speed.

        Args:
            output_path: Path to save quantized model
            quantization_type: Type of quantization ('dynamic' or 'float16')

        Returns:
            Path to quantized model
        """
        print(f"Quantizing model with {quantization_type} quantization...")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantization_type == 'dynamic':
            # Dynamic range quantization (int8 weights, float32 activations)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization_type == 'float16':
            # Float16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

        # Convert the model
        tflite_model = converter.convert()

        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        # Print size comparison
        original_size = os.path.getsize(self.model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100

        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {reduction:.2f}%")

        return output_path

    def benchmark_inference(
        self,
        test_sequences: np.ndarray,
        num_iterations: int = 100
    ) -> dict:
        """
        Benchmark inference performance.

        Args:
            test_sequences: Test sequences for benchmarking
            num_iterations: Number of iterations for benchmarking

        Returns:
            Dictionary with benchmark results
        """
        print(f"\nBenchmarking inference over {num_iterations} iterations...")

        latencies = []

        # Warmup
        for _ in range(10):
            _ = self.model.predict(test_sequences, verbose=0)

        # Benchmark
        for _ in range(num_iterations):
            start_time = time.time()
            _ = self.model.predict(test_sequences, verbose=0)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)

        results = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies)
        }

        print("\nBenchmark Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.2f}")

        return results


class TFLitePredictor:
    """Predictor using TensorFlow Lite quantized model."""

    def __init__(self, model_path: str):
        """
        Initialize TFLite predictor.

        Args:
            model_path: Path to TFLite model
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Predict using TFLite model.

        Args:
            sequences: Input sequences

        Returns:
            Predictions
        """
        predictions = []

        for seq in sequences:
            # Prepare input
            input_data = np.expand_dims(seq, axis=0).astype(np.float32)

            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                input_data
            )

            # Run inference
            self.interpreter.invoke()

            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(output[0])

        return np.array(predictions)

    def benchmark(self, test_sequences: np.ndarray, num_iterations: int = 100) -> dict:
        """
        Benchmark TFLite inference.

        Args:
            test_sequences: Test sequences
            num_iterations: Number of iterations

        Returns:
            Benchmark results
        """
        print(f"\nBenchmarking TFLite inference over {num_iterations} iterations...")

        latencies = []

        # Warmup
        for _ in range(10):
            _ = self.predict(test_sequences)

        # Benchmark
        for _ in range(num_iterations):
            start_time = time.time()
            _ = self.predict(test_sequences)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)

        results = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies)
        }

        print("\nTFLite Benchmark Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.2f}")

        return results


def optimize_and_benchmark(
    model_path: str,
    test_sequences: np.ndarray,
    output_dir: str = 'models'
):
    """
    Complete optimization and benchmarking pipeline.

    Args:
        model_path: Path to Keras model
        test_sequences: Test sequences for benchmarking
        output_dir: Directory to save optimized models
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MODEL OPTIMIZATION AND BENCHMARKING")
    print("=" * 60)

    # Original model benchmarking
    print("\n1. Benchmarking Original Model")
    print("-" * 60)
    optimizer = ModelOptimizer(model_path)
    original_results = optimizer.benchmark_inference(test_sequences)

    # Dynamic quantization
    print("\n2. Dynamic Range Quantization")
    print("-" * 60)
    quantized_path_dynamic = os.path.join(output_dir, 'sentiment_model_quantized_dynamic.tflite')
    optimizer.quantize_model(quantized_path_dynamic, quantization_type='dynamic')

    tflite_predictor_dynamic = TFLitePredictor(quantized_path_dynamic)
    dynamic_results = tflite_predictor_dynamic.benchmark(test_sequences)

    # Float16 quantization
    print("\n3. Float16 Quantization")
    print("-" * 60)
    quantized_path_float16 = os.path.join(output_dir, 'sentiment_model_quantized_float16.tflite')
    optimizer.quantize_model(quantized_path_float16, quantization_type='float16')

    tflite_predictor_float16 = TFLitePredictor(quantized_path_float16)
    float16_results = tflite_predictor_float16.benchmark(test_sequences)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nOriginal Model P99 Latency: {original_results['p99_latency_ms']:.2f} ms")
    print(f"Dynamic Quantized P99 Latency: {dynamic_results['p99_latency_ms']:.2f} ms")
    print(f"Float16 Quantized P99 Latency: {float16_results['p99_latency_ms']:.2f} ms")

    speedup_dynamic = original_results['p99_latency_ms'] / dynamic_results['p99_latency_ms']
    speedup_float16 = original_results['p99_latency_ms'] / float16_results['p99_latency_ms']

    print(f"\nDynamic Quantization Speedup: {speedup_dynamic:.2f}x")
    print(f"Float16 Quantization Speedup: {speedup_float16:.2f}x")


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = 'models/sentiment_model.keras'

    if os.path.exists(MODEL_PATH):
        # Create dummy test sequences
        test_sequences = np.random.randint(0, 10000, size=(10, 100))

        optimize_and_benchmark(MODEL_PATH, test_sequences)
    else:
        print(f"Model not found at {MODEL_PATH}")
        print("Please train the model first using train.py")
