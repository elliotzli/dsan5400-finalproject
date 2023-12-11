import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

origin_df = pd.read_json("../../Data/Software_5.json", lines=True)
df = origin_df[['overall', 'reviewText', 'summary']]

# Convert to lower cases and tokenize
df.loc[:, 'reviewText'] = df['reviewText'].apply(lambda x: str(x).lower())
df.loc[:, 'summary'] = df['summary'].apply(lambda x: str(x).lower())

# Remove special characters
df.loc[:, 'reviewText'] = df['reviewText'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
df.loc[:, 'summary'] = df['summary'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

# Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
df.loc[:, 'reviewText'] = df['reviewText'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stopwords)]))
df.loc[:, 'summary'] = df['summary'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stopwords)]))

# Lemmatization
lemmatizer = nltk.stem.WordNetLemmatizer()
df.loc[:, 'reviewText'] = df['reviewText'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
df.loc[:, 'summary'] = df['summary'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

df = df.rename(columns={'overall': 'rating', 'reviewText': 'review'})

df.head(20)

df.to_csv("../../Data/cleaned_comments.csv", index=False)
df.to_json("../../Data/cleaned_comments.json", orient='records', lines=True, indent=4)