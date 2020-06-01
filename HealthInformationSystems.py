# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from serbian_stemmer import stem

# Importing the dataset
dataset = pd.read_csv("dz nis anamneza i dijagnoza 2018.csv")

# Importing serbian stopwords
stops_words = set(stopwords.words("srpski"))

# Importing unnecessary words
file = open('unnecessary_words.txt', "r")
unnecessary_words = set(file.read().splitlines())
file.close()
stemmed_unnecessary_words = [stem(w) for w in unnecessary_words]

def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X

def clean_text(raw_text):
    
    # Convert to lower case
    text = raw_text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove punctuation
    # use .isalnum if you want to keep numbers also
    token_words = [w for w in tokens if w.isalpha()]
    
    # Remove stop words
    cleaned_words = [w for w in token_words if not w in stops_words]
    
    # Stemming
    stemmed_words = [stem(w) for w in cleaned_words]
    
    #Remove unnecessary words
    cleaned_words = [w for w in stemmed_words if not w in stemmed_unnecessary_words]
    
    #Connect tokens into a sentence
    cleaned_text = ' '.join(cleaned_words)
    
    return cleaned_text

# Get text to clean
text_to_clean = list(dataset['anamneza'])

# Apply cleaning function on the dataset
dataset['anamneza'] = apply_cleaning_function_to_list(text_to_clean)

# Replacing empty string with None
dataset['anamneza'] = dataset['anamneza'].apply(lambda y: np.nan if len(y)==0 else y)
# Removing rows with None in column ['anamneza']
dataset = dataset.dropna(axis = 0, subset = ['anamneza'])

# Creating Bag Of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
# X = sparce_matrix 
X = cv.fit_transform(dataset['anamneza']).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# Dimensionality reduction using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Naive Bayes Classification to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Prediction outcome accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))