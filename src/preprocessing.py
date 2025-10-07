"""
Data preprocessing module for sentiment analysis.
Handles text cleaning, tokenization, and feature extraction.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from typing import List, Tuple

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Handles text preprocessing for sentiment analysis."""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.

        Args:
            text: Raw input text

        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text.

        Args:
            text: Cleaned text string

        Returns:
            List of lemmatized tokens
        """
        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return tokens

    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.

        Args:
            text: Raw input text

        Returns:
            Preprocessed text string
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of raw text strings

        Returns:
            List of preprocessed text strings
        """
        return [self.preprocess(text) for text in texts]


def load_and_preprocess_data(file_path: str) -> Tuple[List[str], List[int]]:
    """
    Load and preprocess data from CSV file.

    Args:
        file_path: Path to CSV file with 'text' and 'sentiment' columns

    Returns:
        Tuple of (preprocessed_texts, labels)
    """
    import pandas as pd

    df = pd.read_csv(file_path)
    preprocessor = TextPreprocessor()

    texts = preprocessor.preprocess_batch(df['text'].tolist())
    labels = df['sentiment'].tolist()

    return texts, labels


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = TextPreprocessor()

    sample_texts = [
        "I love this product! It's amazing!!! 🔥",
        "@user This is terrible. http://example.com #disappointed",
        "Not sure what to think... it's okay I guess"
    ]

    print("Original texts:")
    for text in sample_texts:
        print(f"  - {text}")

    print("\nPreprocessed texts:")
    for text in sample_texts:
        processed = preprocessor.preprocess(text)
        print(f"  - {processed}")
