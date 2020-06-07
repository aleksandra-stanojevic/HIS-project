# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from serbian_stemmer import stem

#importing the data
dataset = pd.read_csv("dz nis anamneza i dijagnoza 2018.csv")

#counting number of rows for each unique value from column 'sifra'
list = dataset.groupby(['sifra']).size().reset_index(name='counts')

#removing rows containing less or equal than 50 in column 'counts'
list = list.drop(list[list.counts < 50].index)

#removing all rows from data which contain a value in column 'sifra' that is in list['sifra']
dataset = dataset[dataset['sifra'].isin(list['sifra'])]

#preparing data for undersampling
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:,[1]].values

# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='not minority')

# fit and apply the transform method
X_over, y_over = undersample.fit_resample(X, y)

X_over = X_over.tolist()
y_over = y_over.tolist()

#turn lists to dataframe
dataset = pd.DataFrame(
    {'anamneza': X_over,
     'sifra': y_over
    })

#convert lists made by undersampling algorithm back to str for NLP processing
strings = []
for text in dataset['anamneza']:
    strings.append(' '.join(map(str, text)))
    
dataset['anamneza'] = strings

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

# Apply cleaning function on the dataset
dataset['anamneza'] = apply_cleaning_function_to_list(dataset['anamneza'])

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