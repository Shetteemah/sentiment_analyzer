# Sentiment Analyzer

A Transformer-based model for sentiment analysis, classifying movie reviews as **Positive** or **Negative** based on text input. Built with TensorFlow and the Hugging Face ecosystem, this project leverages a custom transformer architecture to process text data and predict sentiment, originally designed as a classic use case for Transformers in natural language processing (NLP). The project also utilizes the IMDB dataset for robust training and supports efficient inference on new reviews without retraining.

## Features
- **Dataset**: Utilizes the IMDB movie review dataset (50,000 reviews: 40,000 train, 10,000 test) with a balanced 50% positive/50% negative split for accurate sentiment learning.
- **Preprocessing**: Employs the BERT tokenizer (`bert-base-uncased`) for efficient text tokenization, handling a large vocabulary (~30,000 tokens).
- **Model**: A lightweight custom transformer with 4-head Multi-Head Attention, 64-dimensional embeddings, and Global Average Pooling, optimized for CPU training.
- **Training**: Configured with the Adam optimizer at a low learning rate (1e-6), 10 epochs, batch size of 32, and early stopping to prevent overfitting.
- **Inference**: Supports fast sentiment prediction (~0.03s per review) by loading a saved model, avoiding retraining (~870s).
- **Testing**: Dynamically loads test reviews from `test_reviews.py` for flexible and scalable evaluation.
- **Data Inspection**: Exports the dataset to `imdb_reviews.json` for manual review of raw reviews and labels.

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
    - The `data.py` script loads and preprocesses the IMDB dataset, providing 50,000 labeled reviews (50% positive, 50% negative), and exports to `imdb_reviews.json` for inspection.
    - Run `data.py` independently to verify dataset loading and label distribution:
    ```bash
    python3 data.py

2. **Train the Model**:
- Execute `transformer_model.py` to tokenize the data, train the transformer, and save the model to `sentiment_predictor.h5`:
    ```bash
    python3 transformer_model.py

- Training occurs only if `sentiment_predictor.h5` is absent, taking ~870 seconds (unless interrupted by the `early_stopping()` function) on CPU.
- Expected output includes training/validation metrics and sentiment predictions for sample reviews.

3. **Predict Sentiments**:
- After training, subsequent runs load `sentiment_predictor.h5` for fast inference.
- Test reviews are loaded from `test_reviews.py`, or you can modify it to add new reviews:
    ```bash
    python3 transformer_model.py
- The model predicts sentiments for new reviews. Example:
    ```bash
    Sentiment for 'This film is amazing!': Positive (score: 0.9881)
    Sentiment for 'This film is bad!': Negative (score: 0.0173)

4. **Add New Test Reviews**:
- Feel free to modify `test_reviews.py` to update the sample_reviews list by uncommenting more reviews or with new reviews for testing.
- Example:
    ```python
    sample_reviews = [
        "This movie was incredible!",
        "Worst film ever."
        ]

## Project Structure
- `data.py`: Loads and preprocesses the IMDB dataset, shuffles data, checks label distribution, and exports to `imdb_reviews.json`.
- `transformer_model.py`: Defines the transformer model, tokenizes reviews, trains (if needed), saves/loads the model, and predicts sentiments using test_reviews.py.
- `test_reviews.py`: Contains a list of 19 diverse test reviews for evaluating model performance.
- `sentiment_predictor.h5`: Saved model file (excluded from Git via .gitignore).
- `imdb_reviews.json`: Exported dataset for manual inspection.
- `README.md`: Project documentation (this file).

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Hugging Face `datasets` and `transformers`
- scikit-learn
- numpy

## Example Output
- **Training Run**:
    ```bash
    Total reviews: 50000
    Total labels: 50000
    Positive reviews: 25000 (50.0%)
    Negative reviews: 25000 (50.0%)
    Train sequences shape: (40000, 128)
    Test sequences shape: (10000, 128)
    Epoch 1/10
    1250/1250 [==============================] - 91s 72ms/step - accuracy: 0.7976 - loss: 0.4274 - val_accuracy: 0.8273 - val_loss: 0.3885
    ...
    Saved trained model to sentiment_predictor.h5
    Sentiment for 'I watched this sci-fi blockbuster...': Positive
    Sentiment for 'This romantic comedy was a total letdown...': Negative
    Sentiment for 'This film is amazing!': Positive (score: 0.7500)
    Sentiment for 'This film is bad!': Negative (score: 0.3200)

- **Inference Run**:
    ```bash
    Loading trained model from sentiment_predictor.h5
    Sentiment for 'I watched this sci-fi blockbuster...': Positive
    Sentiment for 'This romantic comedy was a total letdown...': Negative

## License
MIT License (see LICENSE file for details).

## Acknowledgments
- IMDB dataset provided by Hugging Face `datasets`.
- BERT tokenizer from Hugging Face `transformers`.
- Built with TensorFlow for model development.

