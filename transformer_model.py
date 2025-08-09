import tensorflow as tf
import pandas as pd
import numpy as np
from data import reviews, labels

#tokenize: convert text to numbers
vocab = set(" ".join(reviews).split())
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for i, word in enumerate(vocab)}

#convert reviews to sequences of numbers
max_length = max(len(review.split()) for review in reviews)
sequences = [[word_to_id[word] for word in review.split()] for review in reviews]
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding="post")

#convert labels -> numpy array
labels = np.array(labels)

#simple transformer
def create_transformer_model(vocab_size, max_length):
    inputs = tf.keras.layers.Input(shape=(max_length,)) #input layer

    embedding = tf.keras.layers.Embedding(vocab_size, 8)(inputs) #embedding layer: convert nums -> dense vectors

    #transformer layer (simplified with Multi-Head Attention)
    attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=4)(embedding, embedding)
    attention = tf.keras.layers.Dropout(0.1)(attention) #prevent overfitting
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention)

    attention = tf.keras.layers.Flatten()(attention) #Flatten to match Dense input befroe proceeding to output
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(attention) #output layeer (0/1 -ve/+ve)

    #create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

#create and compile model
vocab_size = len(vocab)
model = create_transformer_model(vocab_size, max_length)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#debug
print("padded_sequences shape:", padded_sequences.shape)
print("labels shape:", labels.shape)

#train model
model.fit(padded_sequences, labels, epochs=10, batch_size=2, verbose=1)

#test model
def predict_sentiment(review):
    sequence = [word_to_id.get(word, 0) for word in review.split()] #0 for unknown words
    padded = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length, padding="post")
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

#test with new review
new_review = "This film is amazing!"
print(f"Sentiment for '{new_review}': {predict_sentiment(new_review)}")
