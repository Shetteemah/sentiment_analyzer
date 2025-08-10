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
train_reviews = list(dataset["train"]["text"])
train_labels = list(dataset["train"]["label"]) #1 or 0 +ve/-ve
test_reviews = list(dataset["test"]["text"])
test_labels = list(dataset["test"]["label"])

#The full IMDB 50k dataset seems too large for my hardware (indicated by lack of CUDA drivers).
#I’ll use 10,000 train, 2,000 test to balance performance and memory.
train_reviews = train_reviews[:10000]
train_labels = train_labels[:10000]
test_reviews = test_reviews[:2000]
test_labels = test_labels[:2000]

#combine for model's use
reviews = train_reviews + test_reviews
labels = train_labels + test_labels
