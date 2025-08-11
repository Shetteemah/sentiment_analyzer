from datasets import load_dataset
import numpy as np
from sklearn.utils import shuffle
import json

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
train_reviews = train_reviews[:50000]
train_labels = train_labels[:50000]
test_reviews = test_reviews[:30000]
test_labels = test_labels[:30000]

#shuffle subsets
train_reviews, train_labels = shuffle(train_reviews, train_labels, random_state=42)
test_reviews, test_labels = shuffle(test_reviews, test_labels, random_state=42)

#combine for model's use
reviews = train_reviews + test_reviews
labels = train_labels + test_labels

#convert labels to numpy array
labels = np.array(labels, dtype=np.float32)

#create JSON data structure
json_data = [
    {"review": review, "label": int(label)}
    for review, label in zip(reviews, labels)
]

#save data to JSON file
json_file_path = "imdb_reviews.json"
with open(json_file_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

# Check label distribution
print(f"Total reviews: {len(reviews)}")
print(f"Total labels: {len(labels)}")
print(f"Positive reviews: {np.sum(labels == 1)} ({100 * np.mean(labels == 1):.1f}%)")
print(f"Negative reviews: {np.sum(labels == 0)} ({100 * np.mean(labels == 0):.1f}%)")

# # Print sample reviews and labels
# print("\nSample reviews and labels from JSON:")
# for i in range(2):
#     print(f"Review {i+1}: {json_data[i]['review'][:100]}... (Label: {json_data[i]['label']})")

# print(f"\nSaved {len(json_data)} reviews to {json_file_path}")