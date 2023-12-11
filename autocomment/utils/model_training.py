import pandas as pd
import re
import nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib


df = pd.read_csv('../../Data/cleaned_comments.csv')
df.head(20)

df['review'] = df['summary']+ ' ' +df['review']
df.drop('summary', axis=1, inplace=True)
df = df.dropna()

tfidf_vectorizer_review = TfidfVectorizer(max_features=10000)
tfidf_features_review = tfidf_vectorizer_review.fit_transform(df['review'])
X_train, X_test, y_train, y_test = train_test_split(tfidf_features_review, df['rating'], test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
joblib.dump(nb_model, 'nb_model.joblib')
nb_predictions = nb_model.predict(X_test)
print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))

# SVM Model Hyperparameter Tuning

# param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf', 'poly']}
# grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# best_svm_model = grid_search.best_estimator_
# best_svm_predictions = best_svm_model.predict(X_test)
# print("Best SVM Classification Report:\n", classification_report(y_test, best_svm_predictions))

# print("Best Hyperparameters:", grid_search.best_params_)
# print("Best Model:", grid_search.best_estimator_)

svm_model = SVC(C=10,gamma='scale',kernel='rbf')
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, 'svm_model.joblib')
svm_predictions = svm_model.predict(X_test)
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

# Random Forest Model Hyperparameter Tuning

# param_grid_rf = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30],
#     'min_samples_split': [2, 5, 10]
# }

# grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid_rf, cv=5, n_jobs=-1)
# grid_search_rf.fit(X_train, y_train)
# best_rf_model = grid_search_rf.best_estimator_
# best_rf_predictions = best_rf_model.predict(X_test)
# print("Best Random Forest Classification Report:\n", classification_report(y_test, best_rf_predictions))

# print("Best Hyperparameters:", grid_search_rf.best_params_)
# print("Best Model:", grid_search_rf.best_estimator_)

rf_model = RandomForestClassifier(max_depth=30,n_estimators=300,min_samples_split=5 ,random_state=42,n_jobs=-1)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'rf_model.joblib')
rf_predictions = rf_model.predict(X_test)
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

# KNN Model Hyperparameter Tuning

# param_grid_knn = {
#     'n_neighbors': [3, 5, 7, 9],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }

# grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, n_jobs=-1)
# grid_search_knn.fit(X_train, y_train)
# best_knn_model = grid_search_knn.best_estimator_
# best_knn_predictions = best_knn_model.predict(X_test)
# print("Best KNN Classification Report:\n", classification_report(y_test, best_knn_predictions))

# print("Best Hyperparameters:", grid_search_knn.best_params_)
# print("Best Model:", grid_search_knn.best_estimator_)

knn_model = KNeighborsClassifier(n_neighbors=7,metric='euclidean',weights='distance')
knn_model.fit(X_train, y_train)
joblib.dump(knn_model, 'knn_model.joblib')
knn_predictions = knn_model.predict(X_test)
print("KNN Classification Report:\n", classification_report(y_test, knn_predictions))

# Text Preprocessing

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

text = "Overall, the product is okay. It functions as advertised but doesn't really stand out in any way. The design is pretty standard, and while it works well enough, there are no features that make it particularly exciting or unique. Customer service was fine, nothing too impressive but also not disappointing. It's a decent choice if you need something basic."
cleaned_text = clean_text(text)

new_data_tfidf = tfidf_vectorizer_review.transform([cleaned_text])
nb_prediction = nb_model.predict(new_data_tfidf)
svm_prediction = svm_model.predict(new_data_tfidf)
rf_prediction = rf_model.predict(new_data_tfidf)
knn_prediction = knn_model.predict(new_data_tfidf)
print(nb_prediction,svm_prediction,rf_prediction,knn_prediction)