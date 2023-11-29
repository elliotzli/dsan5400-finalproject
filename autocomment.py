from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Generate a comment
def generate_comment(prompt, max_length=None, temperature=None):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
 
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    # Set default values for max_length and temperature
    max_length = np.random.randint(50, 200)
    temperature = np.random.uniform(0, 1.0)

    # Corrected the usage of np.random and converted to integers
    num_beams = int(np.random.randint(2, 10))
    no_repeat_ngram_size = int(np.random.randint(0, 5))

    # Set num_beam_groups to 1
    num_beam_groups = 1

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beam_groups=num_beam_groups,  # Set num_beam_groups to 1
        early_stopping=True
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def clean_text(text):
    if isinstance(text, list):
        # If the input is a list, clean each element separately
        return [clean_text(element) for element in text]
    else:
        # If the input is a string, apply the cleaning operations
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    
df_reviews = pd.read_csv('cleaned_comments.csv')
df_reviews['review'] = df_reviews['summary']+ ' ' +df_reviews['review']
df_reviews = df_reviews.dropna()
unique_comments = df_reviews['rating'].value_counts()
unique_comments = unique_comments.sort_index()
tfidf_vectorizer_review = TfidfVectorizer(max_features=10000)
tfidf_features_review = tfidf_vectorizer_review.fit_transform(df_reviews['review'])

rating_probabilities = unique_comments / sum(unique_comments)

# List of different prompts
prompts = [
    "this commodity is very good,",
    "I enjoyed using this product,",
    "not satisfied with the quality of",
]

knn_model = joblib.load('model/knn_model.joblib')
nb_model = joblib.load('model/nb_model.joblib')
rf_model = joblib.load('model/rf_model.joblib')
svm_model = joblib.load('model/svm_model.joblib')

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
        generated_comment = generate_comment(random_prompt)
        generated_sentence = generated_comment.split(".")[0] + "."
        print(generated_sentence)
        cleaned_sentence = clean_text(generated_sentence)
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






