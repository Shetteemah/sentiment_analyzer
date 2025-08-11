# Sentiment Analyzer

A Transformer-based model for sentiment analysis, classifying movie reviews as **Positive** or **Negative** based on text input. Built with TensorFlow and the Hugging Face ecosystem, this project leverages a custom transformer architecture to process text data and predict sentiment, originally designed as a classic use case for Transformers in natural language processing (NLP).

## Features
- **Dataset**: Utilizes the IMDB movie review dataset (~12,000 reviews: 10,000 train, 2,000 test) for robust training for model accuracy and generalization.
- **Preprocessing**: Employs the BERT tokenizer (`bert-base-uncased`) for efficient text tokenization, handling a large vocabulary (~30,000 tokens).
- **Model**: A lightweight custom transformer with Multi-Head Attention, embedding layers, and optimized for CPU training.
- **Training**: Configured with a low learning rate (1e-6), 3 epochs, and batch size of 32.
- **Evaluation**: Supports validation on a test set and predicts sentiments for new reviews with raw probability scores for debugging.

<!-- `Unfortunately, it currently returns positive reviews as "Negative". To be fixed.`-->


## Installation
1. **Clone repo**:
   ```bash
   git clone https://github.com/Shetteemah/sentiment_analyzer.git
   cd sentiment_analyzer

2. **Set Up a Virtual Environment (recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
3. **Install Dependencies**:
- Install dependencies from `requirements.txt` (recommended):
    ```bash
    pip install -r requirements.txt

- If `requirements.txt` is missing or outdated, install manually:
    ```bash
    pip install tensorflow datasets transformers scikit-learn numpy

- Note: Ensure `requirements.txt` includes `tensorflow>=2.0`, `datasets`, `transformers`, `scikit-learn`, and `numpy`. Verify compatibility with your Python version (3.8+ recommended).

## Usage
1. **Prepare the Data**:
    - The `data.py` script loads and preprocesses the IMDB dataset, providing 12,000 labeled reviews (50% positive, 50% negative).
    - Run `data.py` independently to verify dataset loading and label distribution:
    ```bash
    python3 data.py

2. **Train the Model**:
- Execute `transformer_model.py` to tokenize the data, train the transformer, and evaluate on a test set:
    ```bash
    python3 transformer_model.py

- Expected output includes training/validation metrics and sentiment predictions for sample reviews.

3. **Predict Sentiments**:
- The model predicts sentiments for new reviews. Example:
    ```bash
    Sentiment for 'This film is amazing!': Positive
    Sentiment for 'This film is bad!': Negative

## Project Structure
- `data.py`: Loads and preprocesses the IMDB dataset, shuffles data, and checks label distribution.
- `transformer_model.py`: Defines the transformer model, tokenizes reviews, trains the model, and provides a `predict_sentiment` function.
- `README.md`: Project documentation (this file).

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Hugging Face `datasets` and `transformers`
- scikit-learn

## Example Output
- **Terminal**:
    ```bash
    Total reviews: 12000
    Total labels: 12000
    Positive reviews: 5998 (50.0%)
    Negative reviews: 6002 (50.0%)
    Train sequences shape: (9600, 64)
    Test sequences shape: (2400, 64)
    Epoch 1/5
    600/600 [==============================] - 15s 25ms/step - accuracy: 0.6523 - loss: 0.6234 - val_accuracy: 0.7890 - val_loss: 0.4512
    ...
    Sentiment for 'This film is amazing!': Positive (score: 0.7500)
    Sentiment for 'This film is bad!': Negative (score: 0.3200)

## License
MIT License (see LICENSE file for details).

## Acknowledgments
- IMDB dataset provided by Hugging Face `datasets`.
- BERT tokenizer from Hugging Face `transformers`.
- Built with TensorFlow for model development.

