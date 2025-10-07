# Dataset Directory

This directory contains datasets for training the sentiment analysis model.

## Dataset Format

The training data should be in CSV format with the following columns:

- `text`: The text to analyze (e.g., tweet, review, comment)
- `sentiment`: The sentiment label as an integer
  - `0`: Negative
  - `1`: Neutral
  - `2`: Positive

## Example Format

```csv
text,sentiment
"I love this product! It's amazing!",2
"This is terrible. Very disappointed.",0
"It's okay, nothing special.",1
```

## Sample Dataset

A sample dataset is provided in `sample_tweets.csv` for testing and demonstration purposes.

## Getting a Real Dataset

To train a production-ready model, you'll need a larger dataset. Here are some popular sentiment analysis datasets:

### 1. Twitter Sentiment Analysis Dataset
- Available on Kaggle: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Contains 1.6 million tweets
- Labels: positive (4) and negative (0)
- You'll need to convert labels to match our format (0, 1, 2)

### 2. Amazon Product Reviews
- Available on Kaggle: [Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- Contains millions of product reviews
- Star ratings can be converted to sentiment labels

### 3. IMDB Movie Reviews
- Available at: [Stanford IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- 50,000 movie reviews
- Binary sentiment (positive/negative)

## Preprocessing Your Data

If you have a dataset in a different format, preprocess it to match the required format:

```python
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Rename columns if needed
df = df.rename(columns={'review': 'text', 'rating': 'sentiment'})

# Convert sentiment labels
# Example: Convert 1-5 star ratings to 0, 1, 2
def convert_sentiment(rating):
    if rating <= 2:
        return 0  # negative
    elif rating == 3:
        return 1  # neutral
    else:
        return 2  # positive

df['sentiment'] = df['sentiment'].apply(convert_sentiment)

# Save in the correct format
df[['text', 'sentiment']].to_csv('data/sentiment_data.csv', index=False)
```

## Using Your Dataset

Once you have your dataset in the correct format:

1. Place it in this directory as `sentiment_data.csv`
2. Run the training script: `python train.py`
3. The model will be trained on your data

## Dataset Size Recommendations

- **Minimum**: 10,000 samples for basic testing
- **Recommended**: 50,000+ samples for good performance
- **Optimal**: 100,000+ samples for production-ready models

The claim of "150K+ tweets" in the README refers to the recommended dataset size for achieving 91%+ accuracy.
