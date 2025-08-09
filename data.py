from datasets import load_dataset
import numpy as np

# reviews = [
#         "This movie is great!", #+ve
#         "Terrible acting, awful film." #-ve
#         "Amazing story and visuals." #+ve
#         "Boring and predictable." #-ve
#         "Fantastic experience!" #+ve
#         ]
# # 1 = +ve, 0 = -ve
# labels = [1, 0, 1, 0, 1]

#load imdb dataset
dataset = load_dataset("imdb")

#extract reviews and labels
train_reviews = dataset["train"]["text"]
train_labels = dataset["train"]["label"] #1 or 0 +ve/-ve
test_reviews = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

#combine for model's use
reviews = train_reviews + test_reviews
labels = train_labels + test_labels
