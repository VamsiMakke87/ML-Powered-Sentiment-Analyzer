"""
LSTM-based sentiment classification model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple, List
import pickle
import os


class SentimentLSTM:
    """LSTM model for sentiment classification."""

    def __init__(
        self,
        vocab_size: int = 10000,
        max_length: int = 100,
        embedding_dim: int = 128,
        lstm_units: int = 64
    ):
        """
        Initialize LSTM sentiment classifier.

        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = None
        self.model = None

    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture.

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length
            ),

            # Spatial Dropout for regularization
            layers.SpatialDropout1D(0.2),

            # Bidirectional LSTM layers
            layers.Bidirectional(
                layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.2)
            ),
            layers.Bidirectional(
                layers.LSTM(self.lstm_units // 2, dropout=0.2)
            ),

            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),

            # Output layer (3 classes: negative, neutral, positive)
            layers.Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def prepare_tokenizer(self, texts: List[str]) -> Tokenizer:
        """
        Prepare tokenizer for text vectorization.

        Args:
            texts: List of text strings

        Returns:
            Fitted tokenizer
        """
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        return self.tokenizer

    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to padded sequences.

        Args:
            texts: List of text strings

        Returns:
            Padded sequences array
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_tokenizer first.")

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        return padded

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return history

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict sentiment for texts.

        Args:
            texts: List of text strings

        Returns:
            Predicted class probabilities
        """
        sequences = self.texts_to_sequences(texts)
        predictions = self.model.predict(sequences)
        return predictions

    def predict_class(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment classes.

        Args:
            texts: List of text strings

        Returns:
            List of sentiment labels
        """
        predictions = self.predict(texts)
        class_indices = np.argmax(predictions, axis=1)

        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return [label_map[idx] for idx in class_indices]

    def save_model(self, model_path: str, tokenizer_path: str):
        """
        Save model and tokenizer.

        Args:
            model_path: Path to save model
            tokenizer_path: Path to save tokenizer
        """
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(model_path)

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units
        }

        config_path = model_path.replace('.keras', '_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

    @classmethod
    def load_model(cls, model_path: str, tokenizer_path: str) -> 'SentimentLSTM':
        """
        Load saved model and tokenizer.

        Args:
            model_path: Path to saved model
            tokenizer_path: Path to saved tokenizer

        Returns:
            Loaded SentimentLSTM instance
        """
        # Load config
        config_path = model_path.replace('.keras', '_config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        # Create instance
        instance = cls(**config)

        # Load model
        instance.model = keras.models.load_model(model_path)

        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            instance.tokenizer = pickle.load(f)

        return instance


if __name__ == "__main__":
    # Test model architecture
    print("Building LSTM model...")
    sentiment_model = SentimentLSTM()
    model = sentiment_model.build_model()
    print("\nModel architecture:")
    model.summary()
