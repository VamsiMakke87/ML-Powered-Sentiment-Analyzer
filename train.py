"""
Training pipeline for sentiment analysis model.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import TextPreprocessor
from src.model import SentimentLSTM


def load_data(file_path: str):
    """
    Load dataset from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with text and sentiment columns
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} samples")
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess text data.

    Args:
        df: DataFrame with 'text' and 'sentiment' columns

    Returns:
        Tuple of (preprocessed_texts, labels)
    """
    print("\nPreprocessing text data...")
    preprocessor = TextPreprocessor()

    texts = preprocessor.preprocess_batch(df['text'].tolist())
    labels = df['sentiment'].tolist()

    print(f"Preprocessed {len(texts)} texts")
    return texts, labels


def prepare_datasets(texts, labels, test_size=0.2, val_size=0.1):
    """
    Split data into train, validation, and test sets.

    Args:
        texts: List of text strings
        labels: List of labels
        test_size: Proportion of test data
        val_size: Proportion of validation data from training data

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\nSplitting data...")

    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
    """
    Train LSTM sentiment model.

    Args:
        X_train: Training texts
        y_train: Training labels
        X_val: Validation texts
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size

    Returns:
        Trained model and history
    """
    print("\nInitializing model...")
    model = SentimentLSTM(
        vocab_size=10000,
        max_length=100,
        embedding_dim=128,
        lstm_units=64
    )

    print("Preparing tokenizer...")
    model.prepare_tokenizer(X_train)

    print("Converting texts to sequences...")
    X_train_seq = model.texts_to_sequences(X_train)
    X_val_seq = model.texts_to_sequences(X_val)

    print("\nBuilding model...")
    model.build_model()
    model.model.summary()

    print(f"\nTraining model for {epochs} epochs...")
    history = model.train(
        X_train_seq, np.array(y_train),
        X_val_seq, np.array(y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        X_test: Test texts
        y_test: Test labels

    Returns:
        Test accuracy and predictions
    """
    print("\nEvaluating model on test set...")

    X_test_seq = model.texts_to_sequences(X_test)
    predictions = model.model.predict(X_test_seq)
    y_pred = np.argmax(predictions, axis=1)

    test_loss, test_acc = model.model.evaluate(X_test_seq, np.array(y_test), verbose=0)

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Negative', 'Neutral', 'Positive']
    ))

    return test_acc, y_pred


def plot_training_history(history):
    """
    Plot training history.

    Args:
        history: Training history object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("\nTraining history plot saved to models/training_history.png")


def main():
    """Main training pipeline."""
    # Configuration
    DATA_PATH = 'data/sentiment_data.csv'
    MODEL_PATH = 'models/sentiment_model.keras'
    TOKENIZER_PATH = 'models/tokenizer.pkl'
    EPOCHS = 10
    BATCH_SIZE = 64

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Load data
    df = load_data(DATA_PATH)

    # Preprocess
    texts, labels = preprocess_data(df)

    # Prepare datasets
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(
        texts, labels
    )

    # Train model
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate
    test_acc, y_pred = evaluate_model(model, X_test, y_test)

    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH, TOKENIZER_PATH)
    print("Model saved successfully!")

    # Test inference
    print("\nTesting inference on sample texts...")
    sample_texts = [
        "I absolutely love this! Best experience ever!",
        "This is terrible. Worst product I've ever used.",
        "It's okay, nothing special really."
    ]

    preprocessor = TextPreprocessor()
    sample_texts_processed = preprocessor.preprocess_batch(sample_texts)
    predictions = model.predict_class(sample_texts_processed)

    print("\nSample predictions:")
    for text, pred in zip(sample_texts, predictions):
        print(f"  Text: {text}")
        print(f"  Sentiment: {pred}\n")


if __name__ == "__main__":
    main()
