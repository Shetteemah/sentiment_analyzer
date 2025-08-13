import tensorflow as tf
import numpy as np
from data import reviews, labels
from test_reviews import sample_reviews
from transformers import BertTokenizer
from tensorflow.keras.callbacks import EarlyStopping
import os

#init BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 128

#tokenize reviews
def tokenize_reviews(reviews, max_length):
    encodings = tokenizer(
        reviews,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf"
    )
    return encodings["input_ids"]

#simple transformer
def create_transformer_model(vocab_size, max_length):
    inputs = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32) #input layer for tokenized ids

    #removed `input_length` after getting UserWarning: "Argument input_length is deprecated. Just remove it."
    embedding = tf.keras.layers.Embedding(vocab_size, 64)(inputs) #embedding layer: convert nums (token IDs) to dense vectors. Increased embedding dim.

    #transformer layer (simplified with Multi-Head Attention)
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(embedding, embedding)
    attention = tf.keras.layers.Dropout(0.1)(attention) #prevent overfitting
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention)

    # #Flatten to match Dense input befroe proceeding to output
    # attention = tf.keras.layers.Flatten()(attention)

    #replaced Flatten with pooling to reduce dimentionality
    pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)

    #dense layers
    dense = tf.keras.layers.Dense(32, activation="relu")(pooled)

    #output
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(dense) #output layeer (0/1 -ve/+ve)

    #create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model_path = "sentiment_predictor.h5"

#check for existing trained model
if os.path.exists(model_path):
    print(f"Loading trained model from {model_path}")
    model = tf.keras.models.load_model(model_path)
else:
    #convert reviews into tokenized sequences
    padded_sequences = tokenize_reviews(reviews, max_length)

    #split into train and test sets e.g 80% train, 20% test
    train_size = int(0.8 * len(reviews))
    train_sequences = padded_sequences[:train_size]
    train_labels = labels[:train_size]
    test_sequences = padded_sequences[train_size:]
    test_labels = labels[train_size:]

    #create and compile model
    vocab_size = tokenizer.vocab_size #BERT tokenizer vocab size ~30k
    model = create_transformer_model(vocab_size, max_length)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    #debug 1 (early stoppage to prevent overfitting)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    #debug 2
    print("\nTrain sequences shape:", train_sequences.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test sequences shape:", test_sequences.shape)
    print("Test labels shape:", test_labels.shape, "\n")

    #train model
    model.fit(
        train_sequences,
        train_labels,
        epochs=10,
        batch_size=32,
        validation_data=(test_sequences, test_labels),
        callbacks=[early_stopping],
        verbose=1
    )

    #save trained model
    model.save(model_path)
    print(f"Saved trained model to {model_path}")

#test model
def predict_sentiment(review):
    encoded = tokenize_reviews([review], max_length)
    prediction = model.predict(encoded, verbose=1)[0][0]
    return "Positive" if prediction > 0.5 else "Negative", prediction

for review in sample_reviews:
    sentiment, score = predict_sentiment(review)
    print(f"Sentiment for '{review}': {sentiment} (score: {score:.4f})")