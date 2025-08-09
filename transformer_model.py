import tensorflow as tf
import pandas as pd
import numpy as np
from data import reviews, labels
from transformers import BertTokenizer

#init BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 128

# #tokenize: convert text to numbers
# vocab = set(" ".join(reviews).split())
# word_to_id = {word: i for i, word in enumerate(vocab)}
# id_to_word = {i: word for i, word in enumerate(vocab)}

# #convert reviews to sequences of numbers
# max_length = max(len(review.split()) for review in reviews)
# sequences = [[word_to_id[word] for word in review.split()] for review in reviews]
# padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding="post")

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

#convert reviews into tokenized sequences
padded_sequences = tokenize_reviews(reviews, max_length)

#convert labels -> numpy array
labels = np.array(labels, dtype=np.float32) #also confirms type

#preview dataset size for verification
print(f"Total reviews: {len(reviews)}")
print(f"Total labels: {len(labels)}")

#split into train and test sets e.g 80% train, 20% test
train_size = int(0.8 * len(reviews))
train_sequences = padded_sequences[:train_size]
train_labels = labels[:train_size]
test_sequences = padded_sequences[train_size:]
test_labels = labels[train_size:]

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

#create and compile model
vocab_size = tokenizer.vocab_size #BERT tokenizer vocab size ~30k
model = create_transformer_model(vocab_size, max_length)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#debug
print("Train sequences shape:", train_sequences.shape)
print("Train labels shape:", train_labels.shape)
print("Test sequences shape:", test_sequences.shape)
print("Test labels shape:", test_labels.shape)

#train model
model.fit(
    train_sequences, 
    train_labels, 
    epochs=5, 
    batch_size=16, 
    verbose=1
)

#test model
def predict_sentiment(review):
    # sequence = [word_to_id.get(word, 0) for word in review.split()] #0 for unknown words
    # padded = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length, padding="post")
    encoded = tokenize_reviews([review], max_length)
    prediction = model.predict(encoded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

#test with new review
new_review = "This film is amazing!"
print(f"Sentiment for '{new_review}': {predict_sentiment(new_review)}")
