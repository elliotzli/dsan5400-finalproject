from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
sys.path.append(os.getcwd())
from autocomment.utils.generate_reviews import GPT2Generator


Generate = GPT2Generator()

df_reviews = pd.read_csv("cleaned_comments.csv")
df_reviews["review"] = df_reviews["summary"] + " " + df_reviews["review"]
df_reviews = df_reviews.dropna()
unique_comments = df_reviews["rating"].value_counts()
unique_comments = unique_comments.sort_index()
tfidf_vectorizer_review = TfidfVectorizer(max_features=10000)
tfidf_features_review = tfidf_vectorizer_review.fit_transform(df_reviews["review"])

rating_probabilities = unique_comments / sum(unique_comments)

# List of different prompts
prompts = [
    "this commodity is very good,",
    "I enjoyed using this product,",
    "not satisfied with the quality of",
]

knn_model = joblib.load("model/knn_model.joblib")
nb_model = joblib.load("model/nb_model.joblib")
rf_model = joblib.load("model/rf_model.joblib")
svm_model = joblib.load("model/svm_model.joblib")

# Define the desired distribution of ratings
rating_distribution = [0.11, 0.05, 0.12, 0.24, 0.48]

# Calculate the number of comments for each rating category
total_comments = 100
rating_counts = np.round(np.array(rating_distribution) * total_comments).astype(int)

# Your existing code
comments = []
ratings = []

# Counter for each rating category
rating_counters = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

# Loop through each rating category
for assigned_rating, count in zip(range(1, 6), rating_counts):
    # Generate comments for the specified count
    for _ in range(count):
        random_prompt = np.random.choice(prompts)
        generated_comment = Generate.generate_reviews_forGPT(random_prompt)
        generated_sentence = generated_comment.split(".")[0] + "."
        print(generated_sentence)
        cleaned_sentence = Generate.clean_text(generated_sentence)
        new_data_tfidf = tfidf_vectorizer_review.transform([cleaned_sentence])
        knn_pred = knn_model.predict(new_data_tfidf)
        nb_pred = nb_model.predict(new_data_tfidf)
        rf_pred = rf_model.predict(new_data_tfidf)
        svm_pred = svm_model.predict(new_data_tfidf)

        # Calculate the mean of predictions
        mean_prediction = np.mean([knn_pred, nb_pred, rf_pred, svm_pred])

        # Check if the count for the rating category is not exceeded
        if rating_counters[assigned_rating] < count:
            # Append the assigned rating
            ratings.append(assigned_rating)
            comments.append(generated_sentence)

            # Increment the counter for the rating category
            rating_counters[assigned_rating] += 1


auto_comments = np.random.choice(comments, size=total_comments, replace=False)
print(auto_comments)
